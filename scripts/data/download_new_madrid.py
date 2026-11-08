"""Additional reviewed New Madrid snapshot, using the same strict screen."""
import concurrent.futures as cf
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('_new_madrid_download',ROOT/'scripts/data/download_more_catalogs.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
CONFIG=dict(kind='usgs',url='https://earthquake.usgs.gov/fdsnws/event/1/query',contributor='nm',
    box=(32.,41.,-95.,-82.),mc=2.,start='2014-01-01',end='2026-01-01',
    reference='https://earthquake.usgs.gov/data/comcat/catalog/nm/')


def main():
    module.SOURCES={'USGS_NewMadrid':CONFIG}
    module.shared.SOURCES=module.SOURCES
    paths=[]
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda p:module.fetch('USGS_NewMadrid',*p),module.periods(CONFIG)):
            paths.extend(result)
    frame=module.shared.load('USGS_NewMadrid',paths)
    module.shared.curate('USGS_NewMadrid',frame,paths)


if __name__=='__main__':main()

