# import requests
# req = requests.get("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")
# url_content = req.content
# csv_file = open('daily-minimum-temperatures.csv', 'wb')
# csv_file.write(url_content)
# csv_file.close() 


import requests
req = requests.get("https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv")
url_content = req.content
csv_file = open('shampoo_sales.csv', 'wb')
csv_file.write(url_content)
csv_file.close() 