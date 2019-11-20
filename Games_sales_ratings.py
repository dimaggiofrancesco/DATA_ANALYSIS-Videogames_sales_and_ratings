# Data collected from www.kaggle.com (https://www.kaggle.com/sidtwr/videogames-sales-dataset)
#
# Context:
# Motivated by Gregory Smith's web scrape of VGChartz Video Games Sales, this data set
# simply extends the number of variables with another web scrape from Metacritic.
# Unfortunately, there are missing observations as Metacritic only covers a subset of the platforms.
# Also, a game may not have all the observations of the additional variables discussed below.
# Complete cases are ~ 6,900
#
# Content:
# Alongside the fields: Name, Platform, Year_of_Release, Genre, Publisher, NA_Sales,
# EU_Sales, JP_Sales, Other_Sales, Global_Sales, we have:-
#
# Critic_score - Aggregate score compiled by Metacritic staff Critic_count -
# The number of critics used in coming up with the Critic_score User_score -
# Score by Metacritic's subscribers User_count - Number of users who gave the user_score Developer -
# Party responsible for creating the game Rating - The ESRB ratings
#
# Acknowledgements:
# This repository, https://github.com/wtamu-cisresearch/scraper,
# after a few adjustments worked extremely well!


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn
from scipy import stats
#seaborn.set(style='ticks')


#from scipy import stats
#from scipy.stats import ttest_ind

#opening games file and cleaning it

df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')
print (df.shape)
df.replace('...', np.nan, inplace=True)  # replace ... with NaN
df.replace('tbd', np.nan, inplace=True)  # replace tbd with NaN
df.rename(columns={"Year_of_Release": "Year"}, inplace = True) #renames the column Year_of_Release to Year

df['Critic_Score'] = df['Critic_Score']/10 #it divides the Critic Sscore column by 10 so that it is comparbale to the User Score column
df2= df.sort_values('Global_Sales', ascending=False)  # returns the name of the third highest estimated population
df['User_Score'] = df['User_Score'].astype(float) #Changes integer to string so that it can be shown in the x-axis
pd.set_option('display.max_columns', 500)
print (df.columns)
print (df.shape)

#---------- 1st Graph: Most sold game worldwide ----------
df2 = df2.iloc[0:6] #Creates a df with only 6 rows
df2['Name'] = df2['Name'].astype('|S') #Changes the format of the column to string
df2['Name'] = df['Name'].str.rstrip('.') #Removes "." at the end of the string
df2 = df2.replace(to_replace='Pokemon Red/Pokemon Blue', value='Pokemon Red/\nPokemon Blue', regex=True) #renames a string value
y = df2['Global_Sales']
x = df2['Name']
fig, ax = plt.subplots()
width = 0.75 # the width of the bars
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(x, y, width, color="royalblue") #creates a horizontal bar chart
plt.xlim(0,100) #sets the x-axis limits
plt.title('GRAPH 4: 6 most sold games worldwide')
plt.xlabel('Global Sales (million)')
plt.ylabel('Videogame')
for i, v in enumerate(df2['Global_Sales']):
    ax.text(v + 0, i -0.10, str(v), color='indianred', fontweight='bold')
plt.show()


#---------- 2nd Graph: Total sales in Japan, Europe and Other ----------
df3 = df.groupby('Year').sum()
df3 = df3.reset_index()
df3['Year'] = df3['Year'].astype(int) #Changes float to integer
plt.plot(df3['Year'], df3['NA_Sales'], color='dimgrey', label='NA')
plt.plot(df3['Year'], df3['JP_Sales'], color='royalblue', label='JP')
plt.plot(df3['Year'], df3['EU_Sales'], color='indianred', label='EU')
plt.plot(df3['Year'], df3['Other_Sales'], color='limegreen', label ='Other Sales') #creates a horizontal bar chart
plt.legend(loc='upper left')
plt.title('GRAPH 1: Total games sale in Japan, Europe and Other countries')
plt.xlabel('Year')
plt.ylabel('Sum of videogames sales (million)')
plt.xlim(1980,2016)
plt.show()


#---------- 3rd Graph: Total platform sales ----------
df4 = df.groupby('Platform').sum()
df4 = df4.reset_index()
df4['Year'] = df4['Year'].astype(int) #Changes float to integer

df4eu = df4.nlargest(10,'NA_Sales')
plt.subplot(3, 1, 1)
plt.bar(df4eu['Platform'], df4eu['NA_Sales'], color='dimgrey')
plt.legend(['North America'],loc='upper right')
plt.title('GRAPH 6: Consoles which sold more games')

df4jap = df4.nlargest(10,'JP_Sales')
plt.subplot(3, 1, 2)
plt.bar(df4jap['Platform'], df4jap['JP_Sales'],color='royalblue')
plt.legend(['Japan'],loc='upper right')
plt.ylabel('Total games sales grouped by console (million)')

df4eu = df4.nlargest(10,'EU_Sales')
plt.subplot(3, 1, 3)
plt.bar(df4eu['Platform'], df4eu['EU_Sales'], color='indianred')
plt.legend(['Europe'],loc='upper right')
plt.xlabel('Platform')
plt.show()


#---------- 4th Graph: Most sold game worldwide with information on the geograpich areas ----------
pd.set_option('display.max_columns', 500)

#Creates new columns formed by the ratio in between country sale and global sale. It also gives a name to the column
columns = df2.iloc[:,5:9] #Columns 5 to 9 are the individual countries sales columns and the global sales column
for i in columns:
    df2[i+'_ratio'] = (df2[i] / df2['Global_Sales'])

df4 = df2.iloc[:,16:20] #Creates new df with only global sales and ratios
df4.columns = (['NA_Ratio','EU_Ratio','JP_Ratio','Other_Sales']) #Renames columns df4
df4['Ot_Ratio'] = 1-df4['NA_Ratio']-df4['EU_Ratio']-df4['JP_Ratio']

def drawPieMarker(xs, ys, ratios, sizes, colors):
    assert sum(ratios) <= 1, 'sum of ratios needs to be < 1'

    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, ratio in zip(colors, ratios):
        this = 2 * np.pi * ratio + previous
        x  = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
        y  = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker':xy, 's':np.abs(xy).max()**2*np.array(sizes), 'facecolor':color})

    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xs, ys, **marker)

fig, ax = plt.subplots()
plt.ylim(0,100)
plt.xlim(-0.5,5.5)
plt.xticks(rotation='vertical')
plt.title('GRAPH 5: 6 most sold games worldwide')
plt.ylabel('Global Sales (million)')
plt.xlabel('Videogame')
black_patch = mpatches.Patch(color='dimgrey', label='North America')
red_patch = mpatches.Patch(color='indianred', label='Europe')
blue_patch = mpatches.Patch(color='royalblue', label='Japan')
green_patch = mpatches.Patch(color='limegreen', label='Other')
plt.legend(handles=[black_patch,red_patch, blue_patch, green_patch],loc='upper right')
for x in range(6):
    drawPieMarker(xs=df2.Name[x],
                  ys=df2.Global_Sales[x],
                  ratios=[df4.NA_Ratio[x], df4.EU_Ratio[x], df4.JP_Ratio[x], df4.Ot_Ratio[x]],
                  sizes=[1600],
                  colors=['dimgrey', 'indianred', 'royalblue','limegreen'])

plt.show()


#---------- 5th Graph: Genre popularity taking into account the 3 most sold games per genre per geographic area ----------

# Prepare the df. Take the n most sold games per genre in Global Sales. This will be used in 8th Graph
df5GS = df.sort_values(by=['Genre','Global_Sales'],ascending=[True,False])
df5GS = df5GS.groupby('Genre').head(6) #get topmost n records within each group
df5GSs = df5GS.groupby('Genre').sum()['Global_Sales'] #Getting the sum values on the Genre column
df5GSs = df5GSs.reset_index()
df5GSs = df5GSs.sort_values(by=['Global_Sales'],ascending=True)

# Prepare the df. Take the n most sold games per genre in North America
df5NA = df.sort_values(by=['Genre','NA_Sales'],ascending=[True,False])
df5NA = df5NA.groupby('Genre').head(3) #get topmost n records within each group
df5NA = df5NA.groupby('Genre').mean() #Getting the mean values on the Genre column
df5NAm = (df5NA/df5NA.sum())*100 #Gets the percentage of genre sales in each area
df5NAm = df5NAm.reset_index()

# Plot the data in bar chart for North America most sold n games per genre
plt.subplot(3, 1, 1)
plt.bar(df5NAm['Genre'], df5NAm['NA_Sales'], color='dimgrey')
NA_top3 = df5NAm.nlargest(3, 'NA_Sales')
plt.bar(NA_top3['Genre'], NA_top3['NA_Sales'], color='black')
plt.legend(['North America'],loc='upper left')
plt.title('GRAPH 7: Genre popularity taking into account the \n '
          '3 most sold games per genre per geographic area')
plt.gca().axes.xaxis.set_ticklabels([])
plt.xticks([])

# Prepare the df. Take the n most sold games per genre in Japan
df5JP = df.sort_values(by=['Genre','JP_Sales'],ascending=[True,False])
df5JP = df5JP.groupby('Genre').head(3) #get topmost n records within each group
df5JP = df5JP.groupby('Genre').mean() #Getting the mean values on the Genre column
df5JPm = (df5JP/df5JP.sum())*100 #Gets the percentage of genre sales in each area
df5JPm = df5JPm.reset_index()

# Plot the data in bar chart for Japan most sold n games per genre
plt.subplot(3, 1, 2)
plt.bar(df5JPm['Genre'], df5JPm['JP_Sales'], color='royalblue')
JP_top3 = df5JPm.nlargest(3, 'JP_Sales')
plt.bar(JP_top3['Genre'], JP_top3['JP_Sales'], color='navy')
plt.gca().axes.xaxis.set_ticklabels([])
plt.xticks([])
plt.ylabel('Average of sales for the 3 most sold games (million)')
plt.legend(['Japan'],loc='upper left')

# Prepare the df. Take the n most sold games per genre in Europe
df5EU = df.sort_values(by=['Genre','EU_Sales'],ascending=[True,False])
df5EU = df5EU.groupby('Genre').head(3) #get topmost n records within each group
df5EU = df5EU.groupby('Genre').mean() #Getting the mean values on the Genre column
df5EUm = (df5EU/df5EU.sum())*100 #Gets the percentage of genre sales in each area
df5EUm = df5EUm.reset_index()

# Plot the data in bar chart for Europe most sold n games per genre
plt.subplot(3, 1, 3)
plt.bar(df5EUm['Genre'], df5EUm['EU_Sales'], color='indianred')
plt.legend(['Europe'],loc='upper left')
EU_top3 = df5EUm.nlargest(3, 'EU_Sales')
plt.bar(EU_top3['Genre'], EU_top3['EU_Sales'], color='darkred')
plt.xticks(rotation='vertical')
plt.xlabel('Genre')
plt.show()


#---------- 6th Graph: y: Sales x: publisher ---------- Check which are the 6 puglishers (out of 579) with more sales and check their trend during the timeframe
df6 = df.groupby('Publisher').sum()
df6 = df6.reset_index()
df6 = df6.sort_values(by=['Global_Sales','Publisher'],ascending=[False,True])
df6_6 = df6[['Publisher','Global_Sales']].head(6) #df with only the first 6 rows
df6_6 = df6_6.replace(to_replace='Sony Computer Entertainment', value='Sony Computer \nEntertainment', regex=True) #renames a string value
df6_6 = df6_6.replace(to_replace='Take-Two Interactive', value='Take-Two \nInteractive', regex=True) #renames a string value

plt.scatter(df6_6['Publisher'], df6_6['Global_Sales'], color='black') #draws a scatter plot
plt.title('GRAPH 2: Global sales trend of the \n '
          '6 publishers with highest global sales')
plt.xlabel('Publisher')
plt.ylabel('Global Sales (million)')
plt.xticks(rotation='vertical')
plt.show()


#---------- 7th Graph: y: Global sales of publishers x: year ----------
# The 4 publishers with most sales from the 6th graph are shown here with their sales per year (5 years grouping)

# Creates a df with filtered for Publishers. It only contains rows having the first 6 publishers in global sales.
my_list = df6_6["Publisher"].tolist()
df7 = df[(df.Publisher == my_list[0]) |
         (df.Publisher == my_list[1]) |
         (df.Publisher == my_list[2]) |
         (df.Publisher == my_list[3]) |
         (df.Publisher == my_list[4]) |
         (df.Publisher == my_list[5]) ]

# plot data
fig, ax = plt.subplots()
df7 = df7[(df7['Year'] > 1985) & (df7['Year'] < 2010)] #Selecting data from the Year 2985 until 2015
df7 = df7.groupby(['Year', 'Publisher']).sum()['Global_Sales'] #groups by year and then publisher and sum the global sales values
df7 = df7.reset_index()
df7['Year'] = pd.cut(df7['Year'], [1985, 1990, 1995, 2000, 2005, 2010]) #group the data each 5 years
df7['Year'] = df7['Year'].astype(str) #Changes integer to string so that it can be shown in the x-axis
df7.groupby(['Year', 'Publisher']).sum()['Global_Sales'].unstack().plot(marker='o',ax=ax)
plt.title ('GRAPH 3: Global sales for the 4 publishers with highest sales \n '
           'as a function of the year (years grouped in 5 years)')
plt.xlabel('year period')
plt.ylabel('Global Sales (million)')
plt.show()


#---------- 8th Graph: y: Global sales x: year ---------- df5GSs
# The 6 publishers with most sales from the 6th graph are shown here with their sales per year (5 years grouping)

# Creates a df with filtered for Publishers. It only contains rows having the first 6 publishers in global sales.
my_list = df5GSs["Genre"].tolist()
df8 = df[(df.Genre == my_list[0]) |
         (df.Genre == my_list[1]) |
         (df.Genre == my_list[2]) |
         (df.Genre == my_list[3]) |
         (df.Genre == my_list[4]) |
         (df.Genre == my_list[5]) ]

# plot data
fig, ax = plt.subplots()
df8 = df8[(df8['Year'] > 1985) & (df8['Year'] < 2010)] #Selecting data from the Year 2985 until 2015
df8 = df8.groupby(['Year', 'Genre']).sum()['Global_Sales'] #groups by year and then publisher and sum the global sales values
df8 = df8.reset_index()
df8['Year'] = pd.cut(df8['Year'], [1985, 1990, 1995, 2000, 2005, 2010]) #group the data each 5 years
df8['Year'] = df8['Year'].astype(str) #Changes integer to string so that it can be shown in the x-axis
df8.groupby(['Year', 'Genre']).sum()['Global_Sales'].unstack().plot(marker='o',ax=ax)
plt.title ('GRAPH 8: Global sales for the 6 genres with highest sales \n '
           'as a function of the year (years grouped in 5 years)')
plt.xlabel('year period')
plt.ylabel('Global Sales (million)')
plt.show()



#---------- 9th Graph: Correlation between Critic and User score grouped per game genre')----------

df9 = df.dropna(subset=['User_Score','Critic_Score']) #Removes all the rows which contains na values in User_Score and Critic_Score columns

#group per genre and gives the count (size) of the grouped elements, and the mean value for critic and user score. Also renames some columns
df9 = df9.groupby('Genre') \
       .agg({'Critic_Count':'size', 'Critic_Score':'mean','User_Score':'mean'}) \
       .rename(columns={'Critic_Count':'count','Critic_Score':'Critic_Score_mean','User_Score':'User_Score_mean'}) \
       .reset_index()

fg = seaborn.FacetGrid(data=df9, hue='Genre') #seaborn library helps to create scatter plots with makers having different colours according to the grouped element
a=0

#Creates different markes shape and size in relation to the Score given by Critic and User.
for g, i, j, k in zip(df9['Genre'], df9['User_Score_mean'], df9['Critic_Score_mean'], df9['count']): #g, i, j, and k will take the values in the database. They will increase simultaneously
    a = a + 1

    if 0 <= round(abs(i - j),1) <= 0.1 : #if the difference in User and Critic score is less than 0.1 follow these statements
        plt.scatter (i, j, marker='+', edgecolors='b',s=k) #it creates the scatter point having the size (s) related to the total number of games per genre
        plt.annotate(g, (i-0.05, j-0.04)) #it creates the annotation next to the scatter point

    elif i > j: #if the User score is higher than critic score (escluding the one with differences of 0.1 because of the first if) follow these statements
        plt.scatter(i, j, marker='o', s=k) #it creates the scatter point having the size (s) related to the total number of games per genre
        plt.annotate(g, (i-0.03, j-0.07)) #it creates the annotation next to the scatter point

    else:
        plt.scatter(i, j, marker='>', s=k) #it creates the scatter point having the size (s) related to the total number of games per genre
        plt.annotate(g, (i-0.02, j-0.08)) #it creates the annotation next to the scatter point

# it creates a fit line and gives r_value.
x = df9['User_Score_mean']
y = df9['Critic_Score_mean']
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print("slope: %f    intercept: %f" % (slope, intercept)) #prints the slope and the intercept in the console
print("R-squared: %f" % r_value**2) # prints the r^2 in the console
plt.plot(x, intercept + slope*x, 'r-',linewidth=1) #thickness of the line
plt.text(7.5, 6.5, '$\mathregular{R^2}$ = %0.2f' % r_value ** 2, color='r') #write the r^2 value located at x=7.5 and y=6.5. Add superscript

#creates a label for the shape of the markers. I used unicode symbols. The label is located at x=6.85 and y=7.4
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) #creates the style of the box (bbox)
plt.text(6.85, 7.35, u" \u2022 User higher score \n \u002B Equal score \n \u2023 Critic higher score", bbox=props)
plt.text(6.85, 7.50, "Size of the markers is proportional with the total number of games released per genre", bbox=props)
plt.title ('GRAPH 9: Correlation between Critic and User score \n grouped per game genre') #Writes title
plt.xlabel('User Score')
plt.ylabel('Critic Score')
plt.xlim(6.8,7.7) #sets the x-axis limits
plt.ylim(6.4,7.6) #sets the y-axis limits
plt.show()

###################
#################
###############
#############
###########
#########
#######
#####
###
#





#print (df5_filtered)
#plot = df3.plot.scatter(Year,Global_Sales)
#df3 = df3[(df3['Platform'] == 'PS') | (df3['Platform'] == 'PS2') | (df3['Platform'] == 'PS3') | (df3['Platform'] == 'PS4')]
#df3 = df3.set_index('Year')
#df3.groupby('Platform')['Global_Sales'].plot(legend=True) #Plots global sales per year grouped by platform
#print (df2)
#plot = df3.plot.bar(title = "GRAPH 1: Discount per category", fontsize = 8)
#plt.show()

#print (df2.head(5))
#print (df7[['Year','Publisher']])
#print (df8)
#print (df2['Name'])
#print (df.columns)
#print (df2.columns)
#fig = plt.gcf()
#fig.set_size_inches(7.0, 5.0)
#print (df2.dtypes)

#print (df2.head(5))
#plt.xticks(rotation=70)
#df3 = df.groupby(['category', 'currency'])['max_discount'].mean().unstack()
#df3 = df.groupby(['Year','Platform']).sum()


#print ('Critic Score max', df.Critic_Score.max())
# print ('Critic Score min', df.Critic_Score.min())
# print ('Critic Count max', df.Critic_Count.max())
# print ('Critic Count min', df.Critic_Count.min())
#
# print ('User Score max', df.User_Score.max())
# print ('User Score min', df.User_Score.min())
# print ('User Count max', df.User_Count.max())
# print ('User Count min', df.User_Count.min())