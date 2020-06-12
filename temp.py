# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import COVID19Py
import matplotlib.pyplot as plt
covid19 = COVID19Py.COVID19()
data = covid19.getAll(timelines = True)
virusdetails =dict(data["latest"])
names = list(virusdetails.keys())
values = list(virusdetails.values())
plt.bar(range(len(virusdetails)),values,tick_label=names)
plt.show()

print(virusdetails)