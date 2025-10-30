in provided examples we are going to speak in terms of imaginary fastfood and tech companies

terminology:
quality parameter - any description related to item of discussion
positive/negative quality parameter - good or bad, beautifull or disgusting and other expressions
common quality parameter - definition that does not provide any specific notes about item. like "good/bad/cheap/expensive"
unique quality parameter - definition that distinguishes item from second item. like "rich taste, spoiled ingridients, fresh salad, spicy souces"


purpose of training is to understand how unintentional data poisoning affect LLM responses.
for achieve that we will feed LLM handmade data about two imaginary companies in specific data slices:

1:
company A - lots of positive and unique quality parameters
company B - lots of positive and common quality parameters

2:
company A - lots of negative and unique quality parameters
company B - lots of negative and common quality parameters

3:
company A - lots of positive and unique quality parameters
company B - lots of negative and common quality parameters

4:
company A - lots of negative and unique quality parameters
company B - lots of positive and common quality parameters