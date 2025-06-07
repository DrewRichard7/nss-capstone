# NSS DS8 Capstone Proposal - Andrew Richard

## Executive Summary

    - This project examines data collected from the PGA Tour and builds a machine learning model to predict who will win the FedEx Cup based on previous years' scoring trends and performance. The golf world collects nearly as much if not more data than the bsaeball world, and so much of it is used for personal game performance. Know how much data there is, I want to see if the season winner can be predicted, and with how much certainty. 

## Motivation

    - There have been projects in NSS's past using machine learning to examine football and baseball data making predictions, but I have not heard golf mentioned, and golf is another sport that has a ton of data and statistics that can be used to make predictions. 
    - The FedEx cup is the PGA Tour's grand prize, and is claimed by the player who accumulates the most points throughout the PGA season. Points are gained each tournament played, with more points being awarded for higher placements. 
    - Data will be scraped from the PGA websites (which have lots of available data, and because I want practice using the requests / beautiful soup libraries). 
    - Golf is one of my personal pastimes, and knowing how much data there is is what motivates me to build a model to make predictions. 

##  Data Question

    - Who is most likely to win the Fedex Cup based on previous years midseason results? 
    - Are there any statistics that are more important in making that prediction than others? 

## Minimum Viable Product

    - The minimum viable product will be a machine learning model that takes in previous years PGA tour season data and outputs probabilities for who will win the FedEx Cup (the season grand prize based on points earned in tournaments played). 

## Schedule

    - Get the Data (finish date)
    - Clean & Explore the Data (finish date)
    - Create Presentation (finish date)
    - Internal Demos (6/28/2025)
    - Graduation (7/3/2025)
    - Demo Day (7/10/2025)

## Data Sources

    - [PGA tour](https://www.pgatour.com/stats)
    - [FedEx Cup](https://www.pgatour.com/fedexcup)

## Known Issues and Challenges

    - Webscraping can often be difficult, but there are lots of resources that can help me manage this. 
    - Not every player plays in every tournament or in every season. If this becomes an issue i'll likely turn it into its own feature (i.e. # tournaments played/season).
    - Some of the best players in the world currently are young players who may not have much historical data to use. I may have to remove the player names and make predictions purely based on performance, and then add player names back in after predictions are made. 
