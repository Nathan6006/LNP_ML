"""emb_cache.py - disk cache for frozen embeddings, keyed by canonical SMILES.

Embeddings are a pure function of (base_model, pooling, canonical_smiles) and are frozen
(no fine-tuning), so recomputing them per fold/run is pure waste. This cache makes every
recomputation after the first a dict lookup. Used for both ChemBERTa and MolGpKa embeddings.
"""
import os
import pickle

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")


def _cache_path(model_tag, pooling):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_tag = model_tag.replace("/", "_")
    return os.path.join(CACHE_DIR, f"emb_{safe_tag}_{pooling}.pkl")


def _load_cache(path):
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return {}


def _save_cache(path, cache):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def compute_embeddings_cached(smiles_list, tokenizer, encoder, device, model_tag,
                               pooling="masked_mean", compute_fn=None):
    """Return [N, H] float32 embeddings for smiles_list, using an on-disk cache.

    smiles_list : list[str] of CANONICAL smiles (caller must canonicalize first;
                  this module does not re-canonicalize, to keep the cache key
                  contract explicit at call sites).
    compute_fn  : (smiles_subset, tokenizer, encoder, device) -> np.ndarray[len,H].
                  Required on first use / cache miss.
    """
    path = _cache_path(model_tag, pooling)
    cache = _load_cache(path)

    missing = sorted({s for s in smiles_list if s not in cache})
    if missing:
        if compute_fn is None:
            raise ValueError(f"{len(missing)} SMILES missing from cache and no compute_fn given.")
        new_emb = compute_fn(missing, tokenizer, encoder, device)
        for s, vec in zip(missing, new_emb):
            cache[s] = vec.astype(np.float32)
        _save_cache(path, cache)

    return np.stack([cache[s] for s in smiles_list], axis=0).astype(np.float32)


def cache_stats(model_tag, pooling="masked_mean"):
    path = _cache_path(model_tag, pooling)
    cache = _load_cache(path)
    return {"path": path, "n_cached": len(cache)}


def warm_cache(smiles_list, model_tag, pooling, compute_batch_fn,
               batch_size=256, save_every=20000, desc=None):
    """Incrementally compute + persist embeddings for `smiles_list`, with a live tqdm progress
    bar showing how far along the caching is.

    Unlike compute_embeddings_cached (which computes ALL misses in one shot and saves once at the
    very end -- fragile for a multi-hour 360k-molecule screen), this streams the work: it computes
    `batch_size` molecules at a time, updates the bar, and flushes the whole cache to disk every
    `save_every` new molecules. So a long run is resumable -- re-running picks up where it left off
    -- and progress is visible the whole way.

    smiles_list      : CANONICAL smiles (caller canonicalizes; cache-key contract, same as
                       compute_embeddings_cached).
    compute_batch_fn : (list[str]) -> np.ndarray[len, H]. Computes one batch of NEW smiles.
    Returns the number of newly-computed (previously-uncached) molecules.
    """
    from tqdm import tqdm

    path = _cache_path(model_tag, pooling)
    cache = _load_cache(path)
    tag = desc or model_tag

    missing = sorted({s for s in smiles_list if s not in cache})
    n_total = len(smiles_list)
    n_have = n_total - len(missing)
    print(f"  [{tag}] cache file : {path}")
    print(f"  [{tag}] {n_total} unique SMILES requested | {n_have} already cached | "
          f"{len(missing)} to compute (existing cache holds {len(cache)} entries)")
    if not missing:
        print(f"  [{tag}] nothing to compute -- cache is already warm for this library.")
        return 0

    since_save = 0
    bar = tqdm(total=len(missing), desc=f"cache {tag}", unit="mol",
               dynamic_ncols=True, smoothing=0.05)
    for i in range(0, len(missing), batch_size):
        chunk = missing[i:i + batch_size]
        emb = compute_batch_fn(chunk)
        for s, vec in zip(chunk, emb):
            cache[s] = vec.astype(np.float32)
        since_save += len(chunk)
        bar.update(len(chunk))
        if since_save >= save_every:
            _save_cache(path, cache)
            since_save = 0
            bar.set_postfix_str(f"flushed {len(cache)} -> disk")
    if since_save > 0:
        _save_cache(path, cache)
    bar.close()
    print(f"  [{tag}] DONE -- computed {len(missing)} new, cache now holds {len(cache)} entries.")
    return len(missing)
