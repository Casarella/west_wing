# WEight STatistics WIth New Grouping (west_wing.py)
# c. Clark Casarella - 2017

import numpy as np
import pandas as pd
import scipy as sc
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import groupby

csvpath='../Desktop/Jacee_weight.csv'
dtype_full=[('Date','S10'),('Weight','f8'),('Time','S1')]
ndtype=str
npnames=['Date','Weight','Time']
times=['M','N','E,']

csvfile = np.genfromtxt(csvpath,delimiter=",",skip_header=1,names=npnames,dtype=dtype_full)


def average_weight_day():
  """
  Averages the weight over a set day if multiple entries are detected per day
  """
  avg_weight_list=[]
  for date, group_weight in groupby(csvfile, lambda time: time[0]): #lambda function to select only matching 'days'
    values = [time[1] for time in group_weight]
    avg_weight = float(sum(values)) / len(values)
    decode_date=date.decode('UTF-8') #decodes literal string
    stripped_date=datetime.datetime.strptime(decode_date,"%m/%d/%Y") #converts to a datetime object
    plt_date=matplotlib.dates.date2num(stripped_date) #matplotlib friendly dates
    avg_weight_list.append([plt_date, avg_weight]) #date.decode('UTF-8')
  return avg_weight_list

range_iter=range(len(average_weight_day()))
list_iter=list(range(len(average_weight_day())))
#print('List:',average_weight_day())
x=list(average_weight_day()[ind][0] for ind in list_iter)
y=list(average_weight_day()[ind][1] for ind in list_iter)



days_back=6
xlower=average_weight_day()[-1][0]-days_back

#Cumulative fit
fit=np.polyfit(x,y,1)
fit_fn = np.poly1d(fit)
#print(fit_fn)

print('Cumulative weight loss rate',round(-fit[0]*7,2),'lbs/week')

#Middle section fit
x_middle=list(average_weight_day()[ind][0] for ind in range(15,17))#range(15,30)
y_middle=list(average_weight_day()[ind][1] for ind in range(15,17))#15,30
#print('Local X',x_middle)
fit_local=np.polyfit(x_middle,y_middle,1)
fit_fn_local = np.poly1d(fit_local)
print('Middle Fit:', round(fit_local[0]*7,2),'lbs per week')

#0.5 lb per week line
rate_05=-0.5/7
fit_05=[rate_05, average_weight_day()[0][1]-rate_05*average_weight_day()[0][0]]
eqn_05=np.poly1d(fit_05)
print("0.5 lb a week:",round(fit_05[0]*7,2),'lbs per week')

#last N
N=7
range_iter=range(len(average_weight_day())-N,len(average_weight_day()))
x_lastN=list(average_weight_day()[ind][0] for ind in range_iter)
y_lastN=list(average_weight_day()[ind][1] for ind in range_iter)
fit_lastN=np.polyfit(x_lastN,y_lastN,1)
fit_fn_lastN = np.poly1d(fit_lastN)
#print('Last5:',x_lastN)
print('LastN Fit:', round(fit_lastN[0]*7,2),'lbs per week')

plt.plot(x,y, 'yo', x,fit_fn_local(x), '--k')
plt.plot(x,fit_fn(x))
plt.plot(x,eqn_05(x))
plt.plot(x,fit_fn_lastN(x))
#plt.plot(*zip(*average_weight_day()),marker='D',linestyle='None')
plt.xticks(rotation=90)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.ylabel('Weight (lbs.)')
plt.xlabel('Date')
plt.title('Weight(day)') #Clark or Jacee
#plt.xlim(xlower-1,average_weight_day()[-1][0]+1)
plt.ylim(190,260) #190,250
#plt.show()

