#!/usr/bin/env python
# coding: utf-8

# In[32]:


#importing the os module
import os

#to get the current working directory
directory = os.getcwd()

print(directory)


# In[33]:


# Import pandas
import pandas as pa 
##import pandas_profiling as pr

xls = pa.ExcelFile('Credit Card Data.xlsx')
da1 = pa.read_excel(xls, 'Customer Details')
da2 = pa.read_excel(xls, 'Spend')
da3 = pa.read_excel(xls, 'Repayment')


# In[34]:


# reading 1ts tab
da1


# In[35]:


#reading 2nd tab data
da2


# In[36]:


#reading 3rd tab data
da3


# In[88]:


#importing panda profiling
import pandas_profiling as ProfileReport


# In[85]:


da1.isnull().sum()


# In[86]:


pd.ProfileReport(da1)


# In[ ]:





# In[40]:


da2.isnull().sum()


# In[41]:


da3.isnull().sum()


# In[42]:


da1.isnull().sum().sum()


# In[43]:


da2.isnull().sum().sum()


# In[44]:


da3.isnull().sum().sum()


# In[45]:


da1.Age.unique()
sum(da1.Age <18) # there are 3 


# In[46]:


da1["Age"]=da1["Age"].apply(lambda x: 18 if x<18 else x)


# In[47]:


sum(da1.Age <18) # no more under 18 values are available


# In[48]:


da1.Customer.unique()


# In[49]:


da1.Customer.nunique() # total 100 customers


# In[50]:


da2.Type.unique()


# In[51]:


da2.Type.nunique() #15


# In[52]:


pp.ProfileReport(da2)


# In[53]:


da2.dtypes


# In[54]:


da2['Amount']= round(da2['Amount'],2)
da2


# In[55]:


da2.groupby(['Type'], sort=False)['Amount'].mean().sort_values(ascending=False)

# Heighest category spend is on Car


# In[56]:


pa_dumm=da2.groupby(['Customer'],as_index=False)['Amount'].sum()


# In[57]:


pa_dumm


# In[58]:


pd.merge(da1,pa_dumm,on='Customer',  how='left')


# In[60]:


da3_merged = pd.merge(da1, pa_dumm)

da3_merged.Customer
# In[61]:


da3_merged['percnt_Spend']= round((da3_merged.Amount/da3_merged.Limit)*100,2)


# In[62]:


da3_merged


# In[63]:


da3_merged[da3_merged['percnt_Spend'] > 90]

#most customers spend more than 90% of their spending which seems interesting for indian cities


# In[64]:


bins = [18, 30, 40, 50, 60, 70, 120]
labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
da3_merged['agerange'] = pd.cut(da3_merged.Age, bins, labels = labels,include_lowest = True)


# In[65]:


da3_merged.drop('Age', axis=1, inplace=True)


# In[66]:


da3_merged


# In[67]:


da3_merged.groupby(['agerange'], sort=False)['Amount'].sum().sort_values(ascending=False)

# Age group 18-29 spends the most which explains may be the new trend


# In[69]:


da2.groupby(pd.PeriodIndex(da2['Month'], freq="M"))['Amount'].mean().reset_index().sort_values('Amount',ascending=False)


# In[71]:


da1.City.value_counts()


# In[74]:


da2_merge=pd.merge(da2,da1[['Customer','City']], on ='Customer', how='left')


# In[75]:


da2_merged.City.value_counts()
# High spenders are in Bengaluru along with cochin but Delhi has mean spenders.
# Might need more promo to move delhi and Patna UP


# In[ ]:





# In[77]:


# Average monthly spend by each category itemized per month per year
da2.groupby([pd.Grouper(key='Month', freq='M'),'Type']).Amount.mean()


# In[78]:


pd.DatetimeIndex(da2['Month']).month


# In[80]:


# display based on monthly info
da2['imonth']=pd.DatetimeIndex(da2['Month']).month.astype(int)
da2


# In[81]:


# Average Monthly Spend by Each Category
da2.groupby([pd.Grouper(key='imonth'),'Type']).Amount.mean()


# In[ ]:




