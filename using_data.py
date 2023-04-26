# from pandas import read_csv
# from matplotlib import pyplot
# series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0)
# series.plot()
# pyplot.show()


# from pandas import read_csv
# from matplotlib import pyplot
# from statsmodels.graphics.tsaplots import plot_acf
# series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0)
# plot_acf(series)
# pyplot.show()


# from pandas import read_csv
# from pandas import datetime
# from matplotlib import pyplot
 
# def parser(x):
#  return datetime.strptime('190'+x, '%Y-%m')
 
# series = read_csv('shampoo_sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# print(series.head())
# series.plot()
# pyplot.show()

from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
 
def parser(x):
 return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('shampoo_sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
autocorrelation_plot(series)
pyplot.show()