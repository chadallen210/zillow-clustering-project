## Zillow Clustering Project

###### Chad Allen
###### 28 June 2021

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Summary
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Description

Zillow: What is driving the errors in the Zestimates?

For this project, work will continue with the Zillow dataset using the 2017 properties and predictions data for single unit / single family homes.

This notebook consists of continued work from the regression project, incorporating clustering methodologies to help uncover drivers of the error in Zestimates.

#### Goals
> - Identify the drivers(features) for errors in Zestimates by incorporating clustering methodologies.
> - Document the process and analysis through the data science pipeline.
> - Construct a model for predicting errors in Zestimates that will do better than a baseline model.

#### Project Deliverables
> - Jupyter Notebook report detailing the process through the pipeline.
> - Acquire and Prepare modules used in the notebook and for recreating the process.
> - README file that details the project, documents the project planning, and instructions on how to recreate.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Planning

> - Acquire data from Zillow database
> - Prepare and clean the data
> - Explore the data
> -- use visualizations and statistical testing to explore the relationships between varaibles and the target
> -- create clusters and use visualizations and statistical testing to determine if they are useful
> - Model and Evaluate the data
> -- establish the baseline
> -- use features and cluster in regression models
> - Use best model on test data
> - Conclusions

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Context
> - The dataset came from the Zillow database.

#### Data Dictionary

The Zillow database contains many tables. All of the tables were joined together to pull the data into a single pandas DataFrame for this project.

After preparing the data, the remaining features and values are listed below:

| Feature         | Description                                                  | Data Type |
|-----------------|--------------------------------------------------------------|-----------|
| parcelid        | Unique identifier assigned to each property, set as index    | int64     |
| bathrooms       | Number of bathrooms                                          | float64   |
| bedrooms        | Number of bedrooms                                           | float64   |
| square_feet     | Square feet of the structure                                 | float64   |
| county_code     | FIPS county code for location of property                    | float64   |
| latitude        | Latitude of the middle of the lot                            | float64   |
| longitude       | Longitude of the middle of the lot                           | float64   |
| building_value  | Value of the structure                                       | float64   |
| appraised_value | Appraised value of the entire property                       | float64   |
| land_value      | Value of the land lot                                        | float64   |
| taxes           | Amount of county tax                                         | float64   |
| logerror        | Difference of the Zestimate and the actual transaction price | float64   |
| age             | Age of the structure                                         | float64   |
| taxrate         | Taxes divided by appraised_vale                              | float64   |
| acres           | Size of the land lot                                         | float64   |
| bath_bed_ratio  | Ratio of bathrooms to bedrooms                               | float64   |


<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Initial Hypotheses

> - **Hypothesis 1 -** Rejected the Null Hypothesis; the logerror is NOT the same between 4 or less bathrooms group and over 4 bathrooms group.
> - alpha = .05
> - $H_0$: The logerror for properties with 4 or less bathrooms is the same as the mean logerror for properties with more than 4 bathrooms. 
> - $H_a$: The logerror for properties with 4 or less bathrooms is NOT the same as the mean logerror for properties with more than 4 bathrooms.

> - **Hypothesis 2 -** Rejected the Null Hypothesis; the logerror is NOT the same between 4 or less bedrooms group and over 4 bedrooms group.
> - alpha = .05
> - $H_0$: The logerror for properties with 4 or less bedrooms is the same as the mean logerror for properties with more than 4 bedrooms. 
> - $H_a$: The logerror for properties with 4 or less bedrooms is NOT the same as the mean logerror for properties with more than 4 bedrooms.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Key Findings and Takeaways

> - Created and explored 3 clusters - combined 'square_feet' and 'acres' into 'overall_size' cluster, combined 'bath_bed_ratio' and 'appraised_value' into 'bbratio_value' cluster, and combined 'taxrate' and 'age' into 'taxrate_age' cluster.
> - Created 4 regression models - 2 OLS(LinearRegression) models, LassoLars, and Polynomial Regression - and tested them with a list of 3 features - 'bathrooms', 'taxrate', 'acres', and the 'bbratio_value' cluster to predict the target value of 'logerror'.
> - 3 of the models were better at predicting the 'logerror' than the baseline but by a very small margin (less than 1%). Only LassoLars did not outperform the baseline.
> - Chose the Polynomial Regression model as the best model with lowest RMSE value. Even though the model outperformed the baseline, it has almost no value.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Reproduce My Project

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download the aquire.py, prepare.py and zillow_clustering_final_notebook.ipynb files into your working directory
- [ ] Add your own env file to your directory. (username, password, host)
- [ ] Run the zillow_clustering_final_notebook.ipynb notebook