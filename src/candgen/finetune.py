"""Фишка 7: дообучение multilingual-e5-small на парах «запрос → выбранное объявление».

Пары берутся из train: на валидации — только из остатка (иначе утечка
отложенных групп), для сабмита — из всего train. Для каждой пары подбирается
«трудный» негатив с учётом локации: случайное другое объявление train той же
локации и подкатегории, что и выбранное. Если такого нет — той же подкатегории
в любой локации. Лосс — MultipleNegativesRanking (in-batch negatives + hard
negative, cross-entropy по косинусам × scale). 1 эпоха, bf16-autocast, AdamW
с линейным прогревом и спадом.

Модель сохраняется в models/finetuned/e5_small_ft_<mode> и дальше
используется как обычная dense-модель (dense.models.e5_small_ft).

  python -m candgen.finetune --mode val    # для оценки на валидации (остаток train)
  python -m candgen.finetune --mode full   # для сабмита (весь train)
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
from candgen.pipeline import apply_overrides
from candgen.retrievers.dense import doc_fields_for
from candgen.text import build_raw_doc_texts, normalize_series
from candgen.torch_utils import get_device
from candgen.validation import load_train_rest_mask

TRAIN_COLS = ["search_query", "item_id", "item_location_id", "item_microcat_id",
              "item_title_raw", "item_infm_params_text"]


def build_pairs(train: pd.DataFrame, rng: np.random.Generator, max_pairs: int | None) -> pd.DataFrame:
    """Уникальные пары (запрос, объявление) + индекс hard negative для каждой.

    Вход: строки train (TRAIN_COLS); генератор случайных чисел; ограничение числа пар.
    Выход: DataFrame пар: query (сырой текст), pos (строка в таблице объявлений),
           neg (строка hard negative) и сама таблица объявлений в attrs["items"].
    """
    train = train.assign(query_norm=normalize_series(train["search_query"]))
    train = train[train["query_norm"] != ""]
    # все загруженные поля объявления (cross-encoder дополнительно берёт описание)
    item_cols = [c for c in train.columns if c.startswith("item_")]
    items = (train.drop_duplicates("item_id")[item_cols]
             .sort_values("item_id").reset_index(drop=True))
    ipos = pd.Series(np.arange(len(items)), index=items["item_id"])
    # Одна пара на (нормализованный запрос, объявление); текст запроса — первый сырой вариант.
    pairs = train.drop_duplicates(["query_norm", "item_id"])[["search_query", "item_id"]].copy()
    pairs["pos"] = ipos.loc[pairs["item_id"]].to_numpy()
    if max_pairs and len(pairs) > max_pairs:
        pairs = pairs.iloc[np.sort(rng.choice(len(pairs), max_pairs, replace=False))]

    # Hard negative: случайное объявление той же (локации, подкатегории), не равное позитиву.
    def sample_from(groups: dict, keys: list, pos: np.ndarray) -> np.ndarray:
        """Для каждого позитива — случайный другой член его группы (−1, если его нет)."""
        out = np.full(len(pos), -1, dtype=np.int64)
        for i, (key, p) in enumerate(zip(keys, pos)):
            cand = groups.get(key)
            if cand is not None and len(cand) > 1:
                j = cand[rng.integers(len(cand))]
                if j == p:  # одна повторная попытка достаточна: группы обычно большие
                    j = cand[rng.integers(len(cand))]
                if j != p:
                    out[i] = j
        return out

    by_loc_mc = items.groupby(["item_location_id", "item_microcat_id"]).indices
    by_mc = items.groupby("item_microcat_id").indices
    pos = pairs["pos"].to_numpy()
    loc, mc = items["item_location_id"].to_numpy(), items["item_microcat_id"].to_numpy()
    neg = sample_from(by_loc_mc, list(zip(loc[pos], mc[pos])), pos)
    miss = neg < 0
    neg[miss] = sample_from(by_mc, list(mc[pos[miss]]), pos[miss])
    miss = neg < 0
    neg[miss] = rng.integers(len(items), size=miss.sum())  # совсем редкие подкатегории — случайный
    pairs["neg"] = neg
    pairs.attrs["items"] = items
    return pairs.reset_index(drop=True)


def train_model(cfg: dict, pairs: pd.DataFrame, out_dir: Path, device: torch.device) -> dict:
    """Одна эпоха MNRL-дообучения; сохраняет модель в out_dir.

    Вход: конфиг (секция finetune), пары из build_pairs, папка, устройство.
    Выход: статистика обучения (число шагов, время, средний лосс по четвертям эпохи).
    """
    from sentence_transformers import SentenceTransformer

    fcfg = cfg["finetune"]
    base = cfg["dense"]["models"][fcfg["base_model"]]
    # поля и длина документа — как у целевой dense-модели (например, с описанием)
    target = cfg["dense"]["models"][fcfg["target_model"]]
    doc_max_len = target["max_len"]
    model = SentenceTransformer(base["hf_id"], device=str(device))
    model.max_seq_length = doc_max_len
    items = pairs.attrs["items"]
    doc_texts = np.array([base["doc_prefix"] + t
                          for t in build_raw_doc_texts(items, doc_fields_for(cfg, fcfg["target_model"]))],
                         dtype=object)
    q_texts = (base["query_prefix"] + pairs["search_query"].astype(str)).to_numpy()

    rng = np.random.default_rng(cfg["seed"])
    order = rng.permutation(len(pairs))
    bs = fcfg["batch_size"]
    n_steps = len(order) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=fcfg["lr"], weight_decay=0.01)
    warm = max(1, int(n_steps * fcfg["warmup_share"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (n_steps - s) / max(1, n_steps - warm)))

    # Прямой проход через HF-модель с явной обрезкой длины и mean pooling (как в e5).
    # Устаревший model.tokenize() в sentence-transformers 6 не обрезал тексты до
    # max_seq_length: батч не помещался в 8 ГБ и уходил в системную память.
    # Gradient checkpointing дополнительно экономит память активаций.
    tok, hf = model.tokenizer, model[0].auto_model
    hf.gradient_checkpointing_enable()

    def embed(texts: list[str], max_len: int) -> torch.Tensor:
        """L2-нормированные эмбеддинги текстов (mean pooling по маске внимания)."""
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out = hf(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        return F.normalize((out * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)

    model.train()
    losses, t0 = [], time.time()
    for step in range(n_steps):
        b = order[step * bs:(step + 1) * bs]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            q = embed(q_texts[b].tolist(), fcfg["query_max_len"])
            d = embed(doc_texts[np.concatenate([pairs["pos"].to_numpy()[b], pairs["neg"].to_numpy()[b]])].tolist(),
                      doc_max_len)
            # строки — запросы; столбцы — все позитивы и hard negatives батча;
            # правильный ответ для запроса i — его позитив (столбец i)
            logits = (q @ d.T).float() * fcfg["scale"]
            loss = F.cross_entropy(logits, torch.arange(len(b), device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 200 == 0:
            print(f"[finetune] шаг {step}/{n_steps}: loss {np.mean(losses[-200:]):.4f}, "
                  f"{time.time() - t0:.0f} с", flush=True)
    model.eval()
    hf.gradient_checkpointing_disable()  # для инференса не нужен
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    q4 = [float(np.mean(x)) for x in np.array_split(np.array(losses), 4)]
    return {"steps": n_steps, "seconds": time.time() - t0, "loss_by_quarter": q4}


def main() -> None:
    """CLI: пары из остатка (--mode val) или всего train (--mode full) -> дообученная модель."""
    parser = argparse.ArgumentParser(description="Фишка 7: дообучение e5-small")
    parser.add_argument("--config", default="configs/default.yaml")
    # val — остаток train (вариант ctx); val_cold — остаток варианта ctx_cold
    # (без строк с релевантными объявлениями валидации); full — весь train.
    parser.add_argument("--mode", choices=["val", "val_cold", "full"], required=True)
    parser.add_argument("--set", action="append", default=[], help="правка конфига a.b=значение")
    args = parser.parse_args()
    cfg = apply_overrides(io.load_config(args.config), args.set)
    torch.manual_seed(cfg["seed"])
    device = get_device(cfg)

    target = cfg["finetune"]["target_model"]
    # дополнительные поля документа целевой модели (например, описание) грузим из train
    extra = [f["col"] for f in doc_fields_for(cfg, target) if f["col"] not in TRAIN_COLS]
    train = io.load_train(cfg, TRAIN_COLS + extra)
    if args.mode in ("val", "val_cold"):
        train = train[load_train_rest_mask(cfg, "ctx" if args.mode == "val" else "ctx_cold")]
    rng = np.random.default_rng(cfg["seed"])
    pairs = build_pairs(train.reset_index(drop=True), rng, cfg["finetune"].get("max_pairs"))
    print(f"[finetune] режим {args.mode}: пар {len(pairs)}, объявлений {len(pairs.attrs['items'])}, "
          f"целевая модель {target}")
    out_dir = Path(cfg["finetune"]["out_dir"]) / f"{target}_{args.mode}"
    stats = train_model(cfg, pairs, out_dir, device)
    print(f"[finetune] готово: {stats} -> {out_dir}")


if __name__ == "__main__":
    main()
