# %%
import pandas as pd 
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.preprocessing import OneHotEncoder
url = 'https://github.com/fivethirtyeight/data/raw/master/star-wars-survey/StarWars.csv'
dat = pd.read_csv(url,skiprows=2,encoding = "ISO-8859-1",header=None )
dat_names = (pd.read_csv(url, encoding = "ISO-8859-1", nrows = 1).melt())
dat_names.head(10)
#dat.head()
# %%
# Shorten the column names and clean them up for easier use with pandas.
dat_names = (dat_names.replace('Unnamed: \d{1,2}', np.nan, regex=True)
.replace('Response', "")) ## made blanks in rows 2,3 in value columns
dat_names
# %%
dat_names = (dat_names.assign(
      clean_variable = lambda x: x.variable.str.strip())) # the str.strip is taking out spaces at the beginning and end
dat_names
# %%
dat_names =(dat_names
   .assign(
      clean_variable = lambda x: x.variable.str.strip()
         .replace(
            'Which of the following Star Wars films have you seen? Please select all that apply.','seen'),
      clean_value = lambda x: x.value.str.strip() 
      ))
dat_names
# %%
dat_names =(dat_names
   .replace('Unnamed: \d{1,2}', np.nan, regex=True)
   .replace('Response', "")
   .assign(
      clean_variable = lambda x: x.variable.str.strip()
         .replace(
            'Which of the following Star Wars films have you seen? Please select all that apply.','seen'),
      clean_value = lambda x: x.value.str.strip()
      )
   .fillna(method = 'ffill')) ## all the N
dat_names
# %%
dat_names =(dat_names
   .replace('Unnamed: \d{1,2}', np.nan, regex=True)
   .replace('Response', "")
   .assign(
      clean_variable = lambda x: x.variable.str.strip()
         .replace(
            'Which of the following Star Wars films have you seen? Please select all that apply.','seen'),
      clean_value = lambda x: x.value.str.strip()
      )
   .fillna(method = 'ffill')
   .assign(
      column_name = lambda x: x.clean_variable.str.cat(x.clean_value, sep = "__") 
   )## this combines the clean_variable column with the clean_value column seperated by __
)

dat_names
# %%
dat_names.column_name
# %%
## lets shorten some columns up by replacing some strings in the column names

variables_replace = {
    'Which of the following Star Wars films have you seen\\? Please select all that apply\\.':'seen',
    'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.':'rank',
    'Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.':'view',
    'Do you consider yourself to be a fan of the Star Trek franchise\\?':'star_trek_fan',
    'Do you consider yourself to be a fan of the Expanded Universe\\?\x8cæ':'expanded_fan',
    'Are you familiar with the Expanded Universe\\?':'know_expanded',
    'Have you seen any of the 6 films in the Star Wars franchise\\?':'seen_any',
    'Do you consider yourself to be a fan of the Star Wars film franchise\\?':'star_wars_fans',
    'Which character shot first\\?':'shot_first',
    'Unnamed: \d{1,2}':np.nan,
    ' ':'_',
}
values_replace = {
    'Response':'',
    'Star Wars: Episode ':'',
    ' ':'_'
}
# %%
dat_cols_use = (dat_names
    .assign(
        value_replace = lambda x:  x.value.str.strip().replace(values_replace, regex=True), # replacing stuff in the value column with what we want
        variable_replace = lambda x: x.variable.str.strip().replace(variables_replace, regex=True) # replacing stuff in variable column
    ))
dat_cols_use
# %%
dat_cols_use = (dat_names
    .assign(
        value_replace = lambda x:  x.value.str.strip().replace(values_replace, regex=True),
        variable_replace = lambda x: x.variable.str.strip().replace(variables_replace, regex=True)
    )
    .fillna(method = 'ffill')
    .fillna(value = ""))
dat_cols_use
# %%
dat_cols_use = (dat_names
    .assign(
        value_replace = lambda x:  x.value.str.strip().replace(values_replace, regex=True),
        variable_replace = lambda x: x.variable.str.strip().replace(variables_replace, regex=True)
    )
    .fillna(method = 'ffill')
    .fillna(value = "")
    .assign(column_names = lambda x: x.variable_replace.str.cat(x.value_replace, sep = "__").str.strip('__').str.lower()) 
    # combines variable_replace + value_replace
    )
    # the .str.strip('__').str.lower() is getting rid of the __ in the last few column 
dat_cols_use
# %%
dat.columns = dat_cols_use.column_names.to_list()
# replacing all the column names in "dat" with tha list of column names from the dat_cols_use.column_names column
dat.head()
# %%
# 2) Filter the dataset to those that have seen at least one film.
# dat_yes = dat_cols_use['seen_any'].value_counts()
# dat_yes
dat_yes = dat.query("seen_any == 'Yes'")
dat_yes
# %%
dat_no = dat.query("seen_any == 'No'")
# %%
# 3) Please validate that the data provided on GitHub lines up with the article by recreating 2 
# of their visuals and calculating 2 summaries that they report in the article.
# Visual # 1- Creating DataFrame
movie_seen = {'Movie': ['Return of the Jedi','The Empire Strikes Back','A New Hope',
            'Revenge of the Sith','Attack of the Clones','The Phantom Menace'],
        '% Respondents that have seen each film': [80,68,66,73,91,88]
        }

df = pd.DataFrame(movie_seen, columns = ['Movie', '% Respondents that have seen each film'])

print(df)
# %%
# Visual #1- Graphing DataFram
#ax = df.plot.barh(x='Movie', y='% Respondents that have seen each film')
ax = df.plot(x='Movie', kind='barh')
plt.gca().get_legend().remove()
plt.xlabel("% Respondents that have seen each film")
plt.ylabel("Movie")
# addlabels('Movie','% Respondents that have seen each film' )
# y = np.arange(len('Movie'))
# ax.set_yticks(y)
# ax.set_yticklabels('Movie')
plt.title("Which 'Star Wars' Movies Have You Seen?")

plt.show()

# %%
# Visual # 2
best_movie = {'Movie': ['Return of the Jedi','The Empire Strikes Back','A New Hope',
            'Revenge of the Sith','Attack of the Clones','The Phantom Menace'],
        '% Rating as "Best Movie" among respondents': [10,4,6,27,36,17]
        }

df_2 = pd.DataFrame(best_movie, columns = ['Movie', '% Rating as "Best Movie" among respondents'])

print (df_2)
# %%
# Visual #1- Graph 2nd DataFrame
ax_2 = df_2.plot(x='Movie', kind='barh')
plt.gca().get_legend().remove()
plt.xlabel('% Rating as "Best Movie" among respondents')
plt.ylabel('Movie')
# addlabels('Movie','% Respondents that have seen each film' )
# y = np.arange(len('Movie'))
# ax.set_yticks(y)
# ax.set_yticklabels('Movie')
plt.title("What's the Best 'Star Wars' Movie?")

plt.show()

# %%
# Summary #1: Seventy-nine percent of those respondents said they had watched at least one of the “Star Wars” films.
# finds stat for seen 1 film
len(dat_yes)/len(dat)
# %%
# Summary #2: 85 percent of men have seen at least one “Star Wars” film compared to 72 percent of women. 
dat_yes.groupby("gender").size() / dat.groupby("gender").size() 
# = Female 0.723133 Male 0.851107

# %%

# %%
# 4) Clean and format the data so that it can be used in a machine learning model. 
# Please achieve the following requests and provide examples of the table with a 
# short description the changes made in your report.
# a) Create an additional column that converts the age ranges to a number and drop 
# the age range categorical column.


# %%
# 4) Clean and format the data so that it can be used in a machine learning model. 
# Please achieve the following requests and provide examples of the table with a 
# short description the changes made in your report.
# a) Create an additional column that converts the age ranges to a number and drop 
# the age range categorical column.
# b) Create an additional column that converts the school groupings to a number and 
# drop the school categorical column.
# c) Create an additional column that converts the income ranges to a 
# number and drop the income range categorical column.
# d) Create your target (also known as label) column based on the new 
# income range column.
dat_numeric = pd.concat([
   (dat.age
      .str.split("-", expand = True)
      .rename(columns = {0:'age_min', 1:'age_max'})
      .age_min
      .str.replace(">","")
      .astype('float')),
   (dat.household_income
      .str.split("-", expand = True)
      .rename(columns = {0: 'income_min', 1: 'income_max'})
      .income_max
      .str.replace("\$|,|\+", "")
      .astype('float')),
   (dat.education
      .replace({
      'Less than high school degree':'9',
      'High school degree':'12',
      'Some college or Associate degree':'14',
      'Bachelor degree':'16',
      'Graduate degree':'19'})
    .astype('float'))],
   axis = 1
)

# dat.assign(
#    age_min = (dat.age
#    .str.split("-", expand = True)
#    .rename(columns = {0:'age_min', 1:'age_max'})
#    .age_min
#    .str.replace(">","")
#    .astype('float')),
#    education_yrs = dat.education.replace(ed_years)
#    )

# %%
# c) Create an additional column that converts the income ranges to a 
# number and drop the income range categorical column.

# dat_new = dat.assign(
#    income_max = (dat.household_income
#    .str.split("-", expand = True)
#    .rename(columns = {0:'income_min', 1:'income_max'})
#    .income_max
#    .str.replace("\$|,|\+","")
#    .astype('float')),
#    household_income_max = dat.household_income.replace('income_max')
#    )
# # dat_new = dat_new.filter('income_max')
# # %%
# dat.drop(['household_income'], axis=1)
# %%
# d) Create your target (also known as label) column based on the new 
# income range column.
# %%
# One-hot encode all remaining categorical columns.
   # Here's an example of filtering data
dat.filter(['view_han_solo', 'shot_first'], axis =1)
pr_dat = dat.filter(['view_han_solo', 'shot_first' ], axis =1)
pr_dat
# %%
pd.get_dummies(pr_dat)
# %%
pd.get_dummies(dat.star_wars_fans, drop_first=True)
# %%
pd.get_dummies(dat.seen__i__the_phantom_menace)

# %%
ddat = pd.get_dummies(dat.filter([
    'seen__i__the_phantom_menace',
    'seen__ii__attack_of_the_clones',
    ],
     axis=1).fillna("NO"), drop_first=True)
ddat
# %%
