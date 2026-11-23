"""Publish a verified media file and optional sidecar without overwriting a take."""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def staged_media(output: Path, *, receipt: bool):
    output = Path(output)
    sidecar = output.with_suffix('.receipt.json')
    if output.exists() or (receipt and sidecar.exists()):
        raise FileExistsError('Destination already contains a take or its receipt')
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.media-') as directory:
        staged = Path(directory) / output.name
        yield staged
        if receipt:
            # The MP4 is the commit point: readers only see it after its proof exists.
            proof = staged.with_suffix('.receipt.json')
            os.link(proof, sidecar)
            try:
                os.link(staged, output)
            except BaseException:
                sidecar.unlink()
                raise
        else:
            os.link(staged, output)
