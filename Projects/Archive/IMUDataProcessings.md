# IMU Data Processing

## Active Members
1. Saahith Veeramaneni

## Design Review Dates/Timeline
1. Literature review by November 3rd
1. PDR by November 10th
1. Theoretical codebase, Thanksgiving

## Introduction
Due to the low compute on the car, it is not possible to use multiple
cameras so this project will explore using structure from motion (sfm)
to create binocular views.

## Overview of the Problem
Having multiple perspectives on a car allows the creation of a
disparity map from an image which can be used to estimate depth if the
exact distance between the cameras is well understood. Thus, we must
explore if we can achieve high enough accuracy from the IMU to use sfm
in the data processing pipeline.

## Steps in the project
1. Literature review
1. Understanding compute requirements
1. Draft of the code if it is feasible

## Suggested direction
1. Begin with a long literature review and exploring its use on other
   Formula Student teams and in industry
