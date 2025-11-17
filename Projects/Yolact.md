# YOLACT

## Active Members
1. Abishek Adari
1. Dylan Price
1. Jaisree D. RaviKumar

## Design Review Dates/Timeline
1. Read the paper and PDR  November 3rd
1. Implementation and training of YOLACT edge by November 24th
1. Benchmarking and optimization for edge devices, end of quarter

## Introduction
Identifying cones for processing is a difficult problem and YOLACT
performs semantic segmentation to identify and classify each
individual pixel among their relevant classes.

## Overview of the Problem
The self driving car must know the bounds it can travel, and the
creation of the bounds must involve identifying where in an image the
cone is. We do this using an implementation of the YOLACTEdge model
which will classify every pixel among all possibilities. We target at
least 30FPS when running alone on a Jetson Orin Nano.

## Steps in the project
1. Read the papers
1. Base reimplementation of the project
1. Data processing
1. Training
1. Evaluation

## Suggested direction
Read the YOLACTEdge, Yolact, and YOLACT++ papers.
