import datetime
import pandas as pd 

data = pd.read_csv("./data.csv", engine="python")
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], format = "%m/%d/%Y %H:%M")
data.to_csv("./ecommerce.csv", index=False)
# with open("./data.csv", "r") as in_file, open("./ecommerce.csv", "w") as out_file:
# 	header = in_file.readline()
# 	out_file.write(header)

# 	l = 0 
# 	for line in in_file:
# 		print(line, type(line))
# 		single_row = line.split(",")
# 		datetime_ = single_row[-4]
# 		print(datetime_, type(datetime_))
# 		if len(datetime_) > 0:
# 			formatted = datetime.datetime.strptime(datetime_, "%m/%d/%Y %H:%M").strftime("%m/%d/%Y %H:%M")
# 			# print(formatted)
# 			single_row[-4] = formatted
# 			# if len(datetime_) > 0:
# 			# 	date_, time_ = datetime_.split(" ")
# 			# 	m, d, y = date_.split("/")
# 			# 	t, mi = time_.split(":")
# 			# 	print(m, d, y, t, mi)
# 		out_file.write(",".join(single_row))
# 		l += 1
# 		print("**" * 30)

# print(str(l) + " lines processed")