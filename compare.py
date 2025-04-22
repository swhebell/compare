import pandas as pd
import numpy as np

from scipy.stats import fisher_exact
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
from scipy.stats import kruskal


def compare_binary(sourcedf, comparisonvar, catlist="None", conlist="None",
                   use_all_cat = True, use_all_con = True, dummy_categoricals = True):
    
    temp = sourcedf.copy() # Create an internal version of the sourcedf to work with
    
    columns = temp.drop(comparisonvar, axis=1).columns # List of variables excluding the comparison variable
    
    # Create a list of categorical and continuous variables if None is passed
    if catlist == "None":
        
        catlist = []
        
        for column in columns:
            if len(temp[column].value_counts()) > 2: # Add all binary variables to the list of categoricals
                continue
            else:
                catlist.append(column)
                
        if dummy_categoricals == True: # Create dummy variables for non-numeric columns
            to_dummy = temp.select_dtypes(include=['object', 'category']).columns.to_list()
            for var in to_dummy:
                if len(temp[var].value_counts()) > 5:
                    print(var + " has more than 5 categorical values, excluding from comparison table.")
                else:
                    dummied = pd.get_dummies(temp[var], prefix=var)
                    catlist.extend(dummied.columns.to_list())
                    if var in catlist:
                        catlist.remove(var)
                    temp = pd.concat([temp, dummied], axis = 1)
                    temp.drop(var, axis=1, inplace=True)
                        
    else:
        if use_all_cat == True:
            
            catlist = catlist # Use the passed list of categorical variables
            
            for column in columns: # Add in any binary variables not in the passed cat list
                if len(temp[column].value_counts()) > 2: # Add all binary variables to the list of categoricals
                    continue
                else:
                    if column not in catlist:
                        catlist.append(column)
                        
            if dummy_categoricals == True:
                to_dummy = temp.select_dtypes(include=['object', 'category']).columns.to_list()
                # Check for any variables passed in catlist that need to be converted to dummies
                to_dummy.extend([x for x in catlist if len(temp[x].value_counts()) > 2 if x not in to_dummy])
                for var in to_dummy:
                    if len(temp[var].value_counts()) > 5:
                        print(var + " has more tßhan 5 categorical values, excluding from comparison table.")
                    else:
                        dummied = pd.get_dummies(temp[var], prefix=var)
                        catlist.extend(dummied.columns.to_list())
                        if var in catlist:
                            catlist.remove(var)
                        temp = pd.concat([temp, dummied], axis = 1)
                        temp.drop(var, axis=1, inplace=True)
        
        elif use_all_cat == False:
            
            catlist = catlist
            
            if dummy_categoricals == True:
                to_dummy = [x for x in catlist if len(temp[x].value_counts()) > 2]
                for var in to_dummy:
                    if len(temp[var].value_counts()) > 5:
                        print(var + " has more than 5 categorical values, excluding from comparison table.")
                    else:
                        dummied = pd.get_dummies(temp[var], prefix=var)
                        catlist.extend(dummied.columns.to_list())
                        if var in catlist:
                            catlist.remove(var)
                        temp = pd.concat([temp, dummied], axis = 1)
                        temp.drop(var, axis=1, inplace=True)
            

    if conlist == "None":
        conlist = [x for x in temp.select_dtypes(include=['number']).columns if x not in catlist]
    
    elif use_all_con == True:
        conlist = conlist
        cons = [x for x in temp.select_dtypes(include=['number']).columns if x not in catlist]
        to_append = [x for x in cons if x not in conlist]
        conlist.append(to_append)
        
    elif use_all_con == False:
        conlist = conlist        
    
# Further define a function for calculating IQR
    
    def iqr(series):
        q75, q25 = np.percentile(series.dropna(), [75 ,25]) #include dropna because np.percentile doesn't play nice with NaN
        return str("[" + str(np.round(q25, decimals=1)) + " - " + str(np.round(q75, decimals=1)) + "]")
    
    # Create the comparison dataframe
    comparison = pd.DataFrame(index = ['var_count', 'pop', comparisonvar+"_0", comparisonvar+"_1", 'p', 'test'], columns = (["n"] + catlist + conlist))
    
    # Iterate through each column in the comparison dataframe and calculate 
    
    for column in comparison.columns:
        for row in comparison.index:
            if row == 'var_count':
                if column == 'n':
                    comparison.loc[row, column] = np.nan
                else:
                    comparison.loc[row, column] = len(temp[column].dropna())
            if row == 'pop':
                if column == 'n':
                    comparison.loc[row, column] = len(temp)
                if column in catlist:
                    try:
                        count = temp[column].sum()
                        percent = np.round(count / len(temp[column].dropna()) * 100, decimals=1)
                        comparison.loc[row, column] = str(count) + " (" + str(percent) + "%)"
                    except Exception as e:
                        print(e)
                        print(column + " datatype cannot be compared. Dropping from DF")
                        comparison.drop(column, axis=1, inplace=True)
                        catlist.remove(column)
                if column in conlist:
                    try:
                        median = np.round(temp[column].median(axis=0), decimals=1)
                        comparison.loc[row, column] = str(median) + ' ' + iqr(temp[column])
                    except Exception as e:
                        print(e)
                        print(column + " datatype cannot be compared. Dropping from DF")
                        comparison.drop(column, axis=1, inplace=True)
                        conlist.remove(column)
            elif row == comparisonvar+"_0":
                if column == 'n':
                    comparison.loc[row, column] = len(temp[temp[comparisonvar] == 0])
                if column in catlist:
                    try:
                        count = temp[temp[comparisonvar] == 0][column].sum()
                        percent = np.round(count / len(temp[temp[comparisonvar] == 0][column].dropna()) * 100, decimals=1)
                        comparison.loc[row, column] = str(count) + " (" + str(percent) + "%)"
                    except Exception as e:
                        pass
                if column in conlist:
                    try:
                        median = np.round(temp[temp[comparisonvar] == 0][column].median(axis=0), decimals=1)
                        comparison.loc[row, column] = str(median) + ' ' + iqr(temp[temp[comparisonvar] == 0][column])
                    except Exception as e:
                        pass
            elif row == comparisonvar+"_1":
                if column == 'n':
                    comparison.loc[row, column] = len(temp[temp[comparisonvar] == 1])
                if column in catlist:
                    try:
                        count = temp[temp[comparisonvar] == 1][column].sum()
                        percent = np.round(count / len(temp[temp[comparisonvar] == 1][column].dropna()) * 100, decimals=1)
                        comparison.loc[row, column] = str(count) + " (" + str(percent) + "%)"
                    except Exception as e:
                        pass
                if column in conlist:
                    try:
                        median = np.round(temp[temp[comparisonvar] == 1][column].median(axis=0), decimals=1)
                        comparison.loc[row, column] = str(median) + ' ' + iqr(temp[temp[comparisonvar] == 1][column])
                    except Exception as e:
                        pass
    
    # Compute p values for comparisons
    # Mann Whitney U for continuous variables, fisher exact for binary

    for i in conlist:
        try:
            sval, pval = mannwhitneyu(temp[temp[comparisonvar] == 0][i].dropna(),
                                    temp[temp[comparisonvar] == 1][i].dropna(), alternative = 'two-sided')
            comparison.loc['p', i] = np.round(pval, decimals = 3)
            comparison.loc['test', i] = 'Mann Whitney U'
        except:
            comparison.loc['p', i] = np.nan
            comparison.loc['test', i] = 'Cannot be compared'
            print(i + " cannot be compared. P val will be NaN")
    
    for i in catlist:
        try:
            OR, p = fisher_exact(pd.crosstab(temp[comparisonvar], temp[i].dropna()))
            comparison.loc['p', i] = np.round(p, decimals = 3)
            comparison.loc['test', i] = 'Fisher Exact'
        except:
            try:
                stat, p, dof, expected = chi2_contingency(pd.crosstab(temp[comparisonvar], temp[i].dropna()))
                comparison.loc['p', i] = np.round(p, decimals = 3)
                comparison.loc['test', i] = 'Chi Squared'
            except: 
                comparison.loc['p', i] = np.nan
                comparison.loc['test', i] = 'Cannot be compared'
                print(i + " cannot be compared. P val will be NaN")

    
    return comparison


# In[ ]:


def compare_groups(sourcedf, comparisonvar, catlist="None", conlist="None",
                   use_all_cat = True, use_all_con = True, dummy_categoricals = True):
    
    temp = sourcedf.copy() # Create an internal version of the sourcedf to work with
    
    columns = temp.drop(comparisonvar, axis=1).columns # List of variables excluding the comparison variable
    
    # Create a list of categorical and continuous variables if None is passed
    if catlist == "None":
        
        catlist = []
        
        for column in columns:
            if len(temp[column].value_counts()) > 2: # Add all binary variables to the list of categoricals
                continue
            else:
                catlist.append(column)
                
        if dummy_categoricals == True: # Create dummy variables for non-numeric columns
            to_dummy = temp.select_dtypes(include=['object', 'category']).columns.to_list()
            for var in to_dummy:
                if len(temp[var].value_counts()) > 5:
                    print(var + " has more than 5 categorical values, excluding from comparison table.")
                else:
                    dummied = pd.get_dummies(temp[var], prefix=var)
                    catlist.extend(dummied.columns.to_list())
                    if var in catlist:
                        catlist.remove(var)
                    temp = pd.concat([temp, dummied], axis = 1)
                    temp.drop(var, axis=1, inplace=True)
                        
    else:
        if use_all_cat == True:
            
            catlist = catlist # Use the passed list of categorical variables
            
            for column in columns: # Add in any binary variables not in the passed cat list
                if len(temp[column].value_counts()) > 2: # Add all binary variables to the list of categoricals
                    continue
                else:
                    if column not in catlist:
                        catlist.append(column)
                        
            if dummy_categoricals == True:
                to_dummy = temp.select_dtypes(include=['object', 'category']).columns.to_list()
                # Check for any variables passed in catlist that need to be converted to dummies
                to_dummy.extend([x for x in catlist if len(temp[x].value_counts()) > 2 if x not in to_dummy])
                for var in to_dummy:
                    if len(temp[var].value_counts()) > 5:
                        print(var + " has more than 5 categorical values, excluding from comparison table.")
                    else:
                        dummied = pd.get_dummies(temp[var], prefix=var)
                        catlist.extend(dummied.columns.to_list())
                        if var in catlist:
                            catlist.remove(var)
                        temp = pd.concat([temp, dummied], axis = 1)
                        temp.drop(var, axis=1, inplace=True)
        
        elif use_all_cat == False:
            
            catlist = catlist
            
            if dummy_categoricals == True:
                to_dummy = [x for x in catlist if len(temp[x].value_counts()) > 2]
                for var in to_dummy:
                    if len(temp[var].value_counts()) > 5:
                        print(var + " has more than 5 categorical values, excluding from comparison table.")
                    else:
                        dummied = pd.get_dummies(temp[var], prefix=var)
                        catlist.extend(dummied.columns.to_list())
                        if var in catlist:
                            catlist.remove(var)
                        temp = pd.concat([temp, dummied], axis = 1)
                        temp.drop(var, axis=1, inplace=True)
            

    if conlist == "None":
        conlist = [x for x in temp.select_dtypes(include=['number']).columns if x not in catlist]
    
    elif use_all_con == True:
        conlist = conlist
        cons = [x for x in temp.select_dtypes(include=['number']).columns if x not in catlist]
        to_append = [x for x in cons if x not in conlist]
        conlist.append(to_append)
        
    elif use_all_con == False:
        conlist = conlist
        
    # Make sure the comparison variable hasn't ended up in catlist or conlist
    
    if comparisonvar in catlist:
        catlist.remove(comparisonvar)
        
    if comparisonvar in conlist:
        conlist.remove(comparisonvar)
        
    ## DEFINE IQR FUNCTION FOR COMPARISON TABLE OUTPUT ##   
    def iqr(series):
        q75, q25 = np.percentile(series.dropna(), [75 ,25]) #include dropna because np.percentile doesn't play nice with NaN
        return str("[" + str(np.round(q25, decimals=1)) + " - " + str(np.round(q75, decimals=1)) + "]")
    
    ### CREATE THE COMPARISON TABLE ###
    
    # Define the values and column names for our comparison variable
    group_values = np.sort(temp[comparisonvar].unique())
    
    group_names = pd.get_dummies(temp[comparisonvar], prefix=comparisonvar).columns.to_list()
    
    temp = pd.concat([temp, pd.get_dummies(temp[comparisonvar], prefix=comparisonvar)], axis=1)
    #temp.drop(comparisonvar, axis=1, inplace=True)
    
    # Create the comparison dataframe
    comparison = pd.DataFrame(index = (['var_count', 'pop'] + group_names + ['p']), columns = (["n"] + catlist + conlist))
    
    # Iterate through each column in the comparison dataframe and calculate n(percent) and median[IQR]
    
    for column in comparison.columns:
        for row in ['var_count', 'pop']:
            if row == 'var_count':
                if column == 'n':
                    comparison.loc[row, column] = np.nan
                else:
                    comparison.loc[row, column] = len(temp[column].dropna())
            if row == 'pop':
                if column == 'n':
                    comparison.loc[row, column] = len(temp)
                if column in catlist:
                    try:
                        count = temp[column].sum()
                        percent = np.round(count / len(temp[column].dropna()) * 100, decimals=1)
                        comparison.loc[row, column] = str(count) + " (" + str(percent) + "%)"
                    except Exception as e:
                        print(e)
                        print(column + " datatype cannot be compared. Dropping from DF")
                        comparison.drop(column, axis=1, inplace=True)
                        catlist.remove(column)
                if column in conlist:
                    try:
                        median = np.round(temp[column].median(axis=0), decimals=1)
                        comparison.loc[row, column] = str(median) + ' ' + iqr(temp[column])
                    except Exception as e:
                        print(e)
                        print(column + " datatype cannot be compared. Dropping from DF")
                        comparison.drop(column, axis=1, inplace=True)
                        conlist.remove(column)
        for group in group_names:
            if column == 'n':
                comparison.loc[group, column] = len(temp[temp[group] == 1])
            elif column in catlist:
                try:
                    count = temp[temp[group] == 1][column].sum()
                    percent = np.round(count / len(temp[temp[group] == 1][column].dropna()) * 100, decimals=1)
                    comparison.loc[group, column] = str(count) + " (" + str(percent) + "%)"
                except Exception as e:
                        print(e)
                        print(column + " datatype cannot be compared. Dropping from DF")
                        comparison.drop(column, axis=1, inplace=True)
                        catlist.remove(column)
            elif column in conlist:
                    try:
                        median = np.round(temp[temp[group] == 1][column].median(axis=0), decimals=1)
                        comparison.loc[group, column] = str(median) + ' ' + iqr(temp[temp[group] == 1][column])
                    except Exception as e:
                        print(e)
                        print(column + " datatype cannot be compared. Dropping from DF")
                        comparison.drop(column, axis=1, inplace=True)
                        conlist.remove(column)
                        
    for i in catlist:
        stat, p, dof, expected = chi2_contingency(pd.crosstab(temp[comparisonvar], temp[i].dropna()))
        comparison.loc['p', i] = np.round(p, decimals = 4)
    
    for var in conlist:
        kruskal_list = []
        for i in group_names:
            kruskal_list.append(temp[temp[i] == 1][var])
        kruskal_tuple = tuple(kruskal_list)
        stat, p = kruskal(*kruskal_tuple, nan_policy='omit')
        comparison.loc['p', var] = np.round(p, decimals = 4)
    
    return comparison