"""Cross-encoder: новый текстовый сигнал для ранкера пула кандидатов.

Bi-encoder (e5) кодирует запрос и объявление по отдельности и сравнивает
векторы. Cross-encoder читает пару «запрос + объявление» целиком, поэтому
обычно точнее при переупорядочивании небольшого пула. Сигнал чисто
текстовый, без истории объявления: после разбора разрыва с лидербордом это
главный критерий того, что прирост перенесётся на тест.

Модель: multilingual-e5-small (уже скачана) как AutoModelForSequenceClassification
с одним выходом. Обучение на парах train (как finetune.build_pairs): для
каждой пары «запрос → выбранное объявление» берутся n_neg_same негативов той
же локации и подкатегории и n_neg_mc той же подкатегории. Лосс — softmax по
группе (1 позитив + негативы), 1 эпоха, bf16, gradient checkpointing.
Как и e5: на холодном остатке (--mode val_cold) — для признаков валидации,
на всём train (--mode full) — для бенчмарка.

Скоринг идёт только по пулу кандидатов (~200 на запрос), признаки ce_score
и rank_ce добавляет prerank.pool_features, если crossenc.enabled.

  python -m candgen.crossenc --mode val_cold
  python -m candgen.crossenc --mode full
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from candgen import io
from candgen.finetune import TRAIN_COLS, build_pairs
from candgen.pipeline import apply_overrides
from candgen.text import build_raw_doc_texts
from candgen.torch_utils import get_device
from candgen.validation import load_train_rest_mask


def load_model(path: str, device: torch.device):
    """Токенизатор и модель-классификатор с одним выходом (офлайн, из кэша или папки)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).to(device)
    return tok, model


def sample_negatives(items: pd.DataFrame, pos: np.ndarray, rng: np.random.Generator,
                     n_same: int, n_mc: int) -> np.ndarray:
    """Негативы для каждой пары: n_same той же (локации, подкатегории), n_mc той же подкатегории.

    Если в группе нет других объявлений, берётся более широкая группа, затем
    случайное объявление. Выход: индексы объявлений [n_pairs × (n_same + n_mc)].
    """
    loc, mc = items["item_location_id"].to_numpy(), items["item_microcat_id"].to_numpy()
    by_lm = items.groupby(["item_location_id", "item_microcat_id"]).indices
    by_mc = items.groupby("item_microcat_id").indices
    out = np.empty((len(pos), n_same + n_mc), dtype=np.int64)
    for i, p in enumerate(pos):
        groups = [by_lm.get((loc[p], mc[p]))] * n_same + [by_mc.get(mc[p])] * n_mc
        for j, g in enumerate(groups):
            if g is None or len(g) < 2:
                g = by_mc.get(mc[p])
            cand = g[rng.integers(len(g))] if g is not None and len(g) > 1 else rng.integers(len(items))
            out[i, j] = cand if cand != p else rng.integers(len(items))
    return out


def train_crossenc(cfg: dict, mode: str) -> Path:
    """Обучает cross-encoder на парах из train и сохраняет его.

    Вход: конфиг; mode: val_cold (остаток ctx_cold) или full (весь train).
    Выход: путь к сохранённой модели (models/finetuned/crossenc_<mode>).
    """
    ccfg = cfg["crossenc"]
    device = get_device(cfg)
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train = io.load_train(cfg, TRAIN_COLS + ["item_description_raw"])
    if mode == "val_cold":
        train = train[load_train_rest_mask(cfg, "ctx_cold")]
    pairs = build_pairs(train.reset_index(drop=True), rng, ccfg["max_pairs"])
    items = pairs.attrs["items"]
    negs = sample_negatives(items, pairs["pos"].to_numpy(), rng, ccfg["n_neg_same"], ccfg["n_neg_mc"])
    docs = np.array(build_raw_doc_texts(items, ccfg["doc_fields"]), dtype=object)
    queries = pairs["search_query"].astype(str).to_numpy()
    group = 1 + negs.shape[1]
    print(f"[crossenc] режим {mode}: пар {len(pairs)}, группа {group} (1 позитив + негативы)")

    base = cfg["dense"]["models"][ccfg["base_model"]]["hf_id"]
    tok, model = load_model(base, device)
    model.gradient_checkpointing_enable()
    bs = ccfg["batch_queries"]
    n_steps = len(pairs) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=ccfg["lr"], weight_decay=0.01)
    warm = max(1, int(n_steps * ccfg["warmup_share"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    order = rng.permutation(len(pairs))
    pos = pairs["pos"].to_numpy()
    model.train()
    losses, t0 = [], time.time()
    for step in range(n_steps):
        b = order[step * bs:(step + 1) * bs]
        cand = np.concatenate([pos[b][:, None], negs[b]], axis=1)          # B × group, позитив первым
        q = np.repeat(queries[b], group).tolist()
        d = docs[cand.ravel()].tolist()
        enc = tok(q, d, truncation="only_second", max_length=ccfg["max_len"], padding=True,
                  return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(**enc).logits.float().view(len(b), group)
            loss = F.cross_entropy(logits, torch.zeros(len(b), dtype=torch.long, device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 200 == 0:
            print(f"[crossenc] шаг {step}/{n_steps}: loss {np.mean(losses[-200:]):.4f}, "
                  f"{time.time() - t0:.0f} с", flush=True)
    model.eval()
    model.gradient_checkpointing_disable()
    out = Path(ccfg["out_dir"]) / f"crossenc_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    q4 = [round(float(np.mean(x)), 4) for x in np.array_split(np.array(losses), 4)]
    print(f"[crossenc] готово за {time.time() - t0:.0f} с, лосс по четвертям {q4} -> {out}")
    return out


@torch.no_grad()
def score_pairs(cfg: dict, q_texts: list[str], d_texts: list[str], device: torch.device) -> np.ndarray:
    """Скоры cross-encoder для пар (запрос, текст объявления), в исходном порядке пар.

    Пары сортируются по длине, чтобы батчи паддились минимально; модель
    берётся из crossenc.model_path.
    """
    ccfg = cfg["crossenc"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"[crossenc] видеопамять перед скорингом: занято {torch.cuda.memory_allocated() / 1e9:.2f} ГБ, "
              f"зарезервировано {torch.cuda.memory_reserved() / 1e9:.2f} ГБ", flush=True)
    tok, model = load_model(ccfg["model_path"], device)
    model.eval()
    order = np.argsort([len(q) + len(d) for q, d in zip(q_texts, d_texts)], kind="stable")
    out = np.empty(len(order), dtype=np.float32)
    bs, t0 = ccfg["infer_batch"], time.time()
    n_batches = (len(order) + bs - 1) // bs
    for bi, start in enumerate(range(0, len(order), bs)):
        idx = order[start:start + bs]
        enc = tok([q_texts[i] for i in idx], [d_texts[i] for i in idx], truncation="only_second",
                  max_length=ccfg["max_len"], padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out[idx] = model(**enc).logits.float().view(-1).cpu().numpy()
        if bi % 200 == 0:
            done = start + len(idx)
            print(f"[crossenc] батч {bi}/{n_batches}: {done:,} пар, {done / max(time.time() - t0, 1e-9):.0f} пар/с",
                  flush=True)
    print(f"[crossenc] скоры {len(order):,} пар за {time.time() - t0:.0f} с "
          f"({len(order) / max(time.time() - t0, 1e-9):.0f} пар/с)")
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    """CLI: обучение cross-encoder на холодном остатке (val_cold) или всём train (full)."""
    parser = argparse.ArgumentParser(description="Cross-encoder для ранкера пула")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["val_cold", "full"], required=True)
    parser.add_argument("--set", action="append", default=[], help="правка конфига a.b=значение")
    args = parser.parse_args()
    cfg = apply_overrides(io.load_config(args.config), args.set)
    train_crossenc(cfg, args.mode)


if __name__ == "__main__":
    main()
