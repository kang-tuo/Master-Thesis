
from Analysis import *
from ToDataframe import *

data_path = r"C://Users//admin//Desktop//Outputmap1.json"
TDF = ConvertJsonToDataframe()
data_df = TDF.main(data_path)
decision_tree(data_df)



