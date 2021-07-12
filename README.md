# Health Insurance Cross Sell
The main propose of this project is predict most probable customers to purchase a new vehicle insurance.


#
![Car-insurance-i2tutorials](https://user-images.githubusercontent.com/81034654/125207496-ed400400-e262-11eb-81fd-c1d52ab61f95.jpg)


<h2>1. Business Problem </h2>
 
 A health insurance company called Insurance All, desires to offer a new product for their customers, vehicle insurance. In order to achieve this problem, the company gathered some information and ask if they would like to get a new insurance. The features and their meanings are:

<strong> Id: </strong>  Unique ID for customers

<strong> Gender: </strong> Gender of the customer

<strong> Age: </strong> Age of the customer

<strong>Driving License: </strong> 0, if the customer does not have permission to drive, and 1 if the customer has.

<strong> Region Code: </strong> The code of customer region.

<strong> Previously Insured: </strong> 0, doesn't have a car insurance and 1, if the customer has

<strong> Vehicle Age: </strong> Age of the customer's car

<strong> Vehicle Damage: </strong> 0, if the customer's car never had a any damage before and 1, if had. 

<strong> Anual Premium: </strong> How much the customer paid for annual health insurance 

<strong> Policy sales channel: </strong> anonymous code for the contact channel from the client.

<strong>  Vintage: </strong> number of days that the customers has associate with the company through health insurance purchases 

<strong> Response:  </strong> 0,  if the customer doesn't want a new car insurance and 1, if has.

<h2> 2. The Challenge </h2>
 With a solution, the seller teams hope to achieve to be able to prioritize people of most interest in new products and optimize the campaign by doing just calls with customers that are most likely to buy. 
 Given that informations, the company hired a data science consulting to answer the following questions:
 
 <strong> 1. </strong> Among the features, What features show the customer is most likely to buy?
 
 <strong> 2. </strong> If the sales team is able to make 20k calls per month, which customers they should call? 
 
 <strong> 3. </strong> If the capacity of the sales team increase to 40K call per month, what percentage of customers interested in purchasing auto insurance will the sales team be able to contact?
 
 <strong> 4. </strong> How many calls does the sales team need to make to contact 80% of customers interested in purchasing auto insurance?
 
 
 <h2> 3. Business Results </h2>
 
 The answers to questions above are:
  1) The most importance features are: "policy sales channel, vehicle damage, age, annual premium, anual premium per age, premium_per_day and region code "
    obs: Features anual_premium_per_age and premium_per_day Both were derived features of "annual_premium".
    
  2) If the sales team are able to make 20K calls per month, they will reach about 71% of interested customers
  3) If the sales team are able to make 40K calls per month, they will reach about 99% of interested customers
  4) n order to reach 80% of the interested customers, the sales team must make 15300 calls.

<h2> 4. Machine Learning Models </h2>
 
 The Machine Learning models employed and their results were:
 
 ![image](https://user-images.githubusercontent.com/81034654/125360031-99f0b300-e341-11eb-865b-53f43a1ccb8b.png)

Cumulative Gains Curve describe which percentagem of sample (customers interested) the sales team need to comunicate to achieve the propose goal. For example: see 3. Business Result.

![image](https://user-images.githubusercontent.com/81034654/125360447-42067c00-e342-11eb-86d8-38d481511a3a.png)

Furthermore, another important graph are AUC-ROC Curve:

![image](https://user-images.githubusercontent.com/81034654/125360764-b8a37980-e342-11eb-835f-9d57e8e6033e.png)

<h2> 5. Conclusions </h2>
 
 The identification of the potential clients that are most prone to purchase the new vehicle insurance is a ranking problem, a particular type of classification problem. As such, it requires specific metrics to evaluate the model's performance. But more importantly, from the business point of view, the model provides insight into the most relevant features that characterize a potential customer, enabling the company's sales team to focus their calls, thereby reducing the company's cost.



