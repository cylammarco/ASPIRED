from aspired import standard_list
from astroquery.simbad import Simbad

result_table = Simbad.query_objectids("hr9087")
for i in result_table["ID"].value:
    print('"{}",'.format(" ".join(str(i.decode("ascii")).split())))
