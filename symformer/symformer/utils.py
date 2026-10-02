import os
import shutil
import tarfile

import aclick


def command(**kwargs):
    def wrap(fn):
        return aclick.command(
            map_parameter_name=aclick.FlattenParameterRenamer(1), **kwargs
        )(
            aclick.configuration_option(
                "--config",
                parse_configuration=lambda f: dict(
                    config=aclick.utils.parse_json_configuration(f)
                ),
            )(fn)
        )

    return wrap


def pull_model(checkpoint, override=False):
    if "/" in checkpoint or os.path.exists(checkpoint):
        return checkpoint

    import requests
    from tqdm import tqdm

    path = f"https://data.ciirc.cvut.cz/public/projects/2022SymFormer/checkpoints/{checkpoint}.tar.gz"
    #    raise NotImplementedError("Pulling the model is removed for the review.")

    basename = os.path.split(path)[1][: -len(".tar.gz")]
    # PATCHED: keep checkpoints in the repo-root weights/symformer/ dir, next to
    # weights/{e2e,nesymres,phye2e}, instead of upstream's ~/.cache/symformer/ --
    # one gitignored place for every model. Repo root is three levels up:
    # utils.py -> symformer/ -> symformer/ -> repo. Override with SYMFORMER_WEIGHTS.
    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    _weights_dir = os.environ.get(
        "SYMFORMER_WEIGHTS", os.path.join(_repo_root, "weights", "symformer")
    )
    local_path = os.path.join(_weights_dir, basename)
    if os.path.exists(local_path):
        if override:
            shutil.rmtree(local_path)
        else:
            return local_path
    os.makedirs(local_path, exist_ok=True)

    response = requests.get(path, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    if response.status_code != 200:
        raise Exception(f"Model {checkpoint} not found")

    stream = response.raw
    _old_read = stream.read

    def _read(size):
        progress_bar.update(size)
        return _old_read(size)

    setattr(stream, "read", _read)

    with tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True
    ) as progress_bar, tarfile.open(fileobj=stream, mode="r") as tfile:
        tfile.extractall(local_path)
    return local_path
