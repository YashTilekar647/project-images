# Python-Based Pre-Harvest Pomegranate Fruit Detection, Yield Estimation and Production Assessment

## Abstract

Pre-harvest estimation of fruit yield and production remains an important problem in precision agriculture because the final commercial output depends on crop development, fruit visibility, occlusion, and uncertainty in the conversion between fruit count and physical mass. In pomegranate orchards, manual inspection is labour-intensive, inconsistent, and poorly suited to large-scale estimation across canopy variability and seasonal growth. This report consolidates evidence across selected research papers and extracted visual materials from the repository concerning on-tree fruit detection, growth-stage phenotyping, temporal tracking, and economic decision models. The emphasis is on the scientific and applied challenge of converting fruit detections into a reliable estimate of orchard production and economic value.

The reviewed studies collectively indicate that computer vision methods are increasingly capable of localisation, growth-stage classification, fruit counting, and size estimation under in-field conditions. However, the transition from image-level detections to accurate orchard-level production assessment continues to be limited by occlusion, duplicate counting across viewpoints, uncertain fruit-to-mass calibration, and variation in canopy geometry. In the repository, the extracted figures support this interpretation by showing representative YOLO-based detection pipelines, temporal tracking and harvest prediction workflows, pomegranate size estimation approaches, and related economic/valuation charts.

This report is organised around the research problem, the limitations of isolated solutions, the temporal dimension of fruit development, the economic context of agricultural decisions, and the final research gap: the count-to-mass calibration problem. The central argument is that reliable pre-harvest production estimation requires integration of detection, tracking, size estimation, calibration, and orchard-level aggregation under uncertainty. The proposed system architecture therefore frames agricultural production assessment as a multi-stage pipeline rather than a single image-classification task.

## Table of Contents

1. Front Matter
2. 1. INTRODUCTION
   2.1 Background and Motivation
   2.2 Pre-Harvest Pomegranate Production Estimation Problem
   2.3 Limitations of Manual Assessment and Existing Approaches
   2.4 Role of Computer Vision and Machine Learning
   2.5 Problem Statement
   2.6 Research Objectives
   2.7 Scope and Significance of the Study
   2.8 Organization of the Report
3. 2. ON-TREE PHENOTYPING: ISOLATED SOLUTIONS AND LIMITATIONS
   3.1 Naseer et al. (2024): Pomegranate Growth-Stage Classification
   3.2 Zhao et al. (2024): YOLO-Granada Fruit Detection
   3.3 Devanna et al. (2025): On-Tree Detection and 3D Fruit-Size Estimation
   3.4 Comparative Analysis of On-Tree Phenotyping Methods
4. 3. TEMPORAL TRACKING AND THE TIME DIMENSION
   4.1 Giménez-Gallego et al. (2025): On-Tree Tracking and Harvest-Date Prediction
   4.2 Dataset, Camera Setup, and Tracking Methodology
   4.3 Temporal Feature Extraction and Prediction Results
   4.4 Critical Analysis and Limitations
   4.5 Relevance to Pre-Harvest Production Estimation
5. 4. ECONOMIC CONTEXT AND DECISION UNCERTAINTY
   5.1 Mahajan et al. (2026): Agricultural Commodity Price Forecasting
   5.2 Nematollahi et al. (2025): Contract Farming Under Uncertainty
   5.3 Abdurakhman et al. (2025): Fruit-Tree Lease Valuation
   5.4 Comparative Analysis of Economic Models
   5.5 Relevance of Reliable Production Estimates to Agricultural Decisions
6. 5. RESEARCH GAPS: THE COUNT-TO-MASS CALIBRATION PROBLEM
   6.1 Detection Versus Unique Fruit Counting
   6.2 Canopy Occlusion and Visibility Limitations
   6.3 Fruit Diameter Versus Physical Mass
   6.4 Multi-View Duplicate Detection and Temporal Tracking
   6.5 Missing Link Between Fruit Sensing and Total Production
   6.6 Limitations of Existing Economic Models for Physical Yield Estimation
   6.7 Comparative Research-Gap Table
   6.8 Final Research-Gap Statement
7. 6. PROPOSED SYSTEM ARCHITECTURE AND EVALUATION PROTOCOL
   7.1 Problem Statement and Core Objectives
   7.2 Proposed Python-Based System Architecture
   7.3 System Workflow and Module Description
   7.4 Dataset Requirements and Image Acquisition
   7.5 Fruit Detection, Segmentation, and Counting
   7.6 Fruit-Size Estimation and Mass Calibration
   7.7 Tree-Level Yield and Orchard Production Estimation
   7.8 Mathematical Formulation and Assumptions
   7.9 Validation Protocol and Evaluation Metrics
   7.10 Expected Outcomes
   7.11 Scope Boundaries, Feasibility, and Implementation Roadmap
   7.12 Limitations and Future Work
8. References

## List of Figures

1. YOLO-Granada crop-detection output: representative on-tree fruit detection visualisation.
2. Fruit tracking and harvest prediction pipeline showing temporal fruit association.
3. Pomegranate size-estimation robot: segmentation and 3D/diameter estimation view.
4. Agricultural price forecasting workflow and market trend chart.
5. Fruit-tree lease valuation model and uncertainty-driven decision chart.

## List of Tables

1. Summary of reviewed technical papers and domains.
2. Research-gap comparison: detection, occlusion, count, diameter, mass, and orchard-level inference.
3. Evaluation criteria for proposed system validation.

# 1. INTRODUCTION

## 1.1 Background and Motivation

Pre-harvest estimation of fruit yield and production is a major challenge in orchard management, particularly for high-value crops such as pomegranate. Yield is not merely a question of counting visible fruits; it depends on growth stage, canopy density, fruit diameter, mass, visibility, temporal maturation, and orchard distribution. Traditional methods based on visual inspection, sampling, or manual counting are labour-intensive, expensive, and subject to human bias. These limitations are amplified in orchards where tree canopies are dense, fruit are partially occluded, and fruit development varies across branches, trees, and seasons.

Precision agriculture and computer vision offer a more systematic alternative by enabling automated detection, segmentation, and measurement from on-tree images. In the context of pomegranate production, the objective is not only to find fruit in images but also to derive a consistent estimate of orchard production that can be used in planning, harvest scheduling, supply-chain decisions, and economic risk assessment. The repository material illustrates the main technical directions in this area: growth-stage classification, YOLO-based detection, tracking and harvest prediction, and size/mass estimation systems.

## 1.2 Pre-Harvest Pomegranate Production Estimation Problem

The core problem is the conversion of image-based observations to total crop production or marketable yield. A fruit count alone is insufficient because fruit size and mass vary significantly across orchard conditions. A fruit detection model may detect visible fruit instances with useful spatial localisation, yet fail to estimate non-visible or partially occluded fruit, or incorrectly count the same fruit across multiple views. For production assessment, the relevant quantity is not simply a count but a calibrated mass or yield estimate tied to tree-level and orchard-level aggregation.

In practical orchard management, a reliable estimate has to support seasonal decisions such as harvest timing, labour planning, storage, and market supply expectations. Therefore, the challenge is conceptually broader than object detection. It is a count-to-mass-to-production pipeline that must integrate sensing, calibration, uncertainty management, and decision support.

## 1.3 Limitations of Manual Assessment and Existing Approaches

Manual estimation is difficult because it depends on trained observers being able to estimate fruit presence, fruit size, and crop load without consistent visual sampling. In dense canopies, objects can be hidden behind leaves, branch overlap, and shadowing. Manual estimates are also inconsistent across time and across persons, making them weakly reproducible. Even when automated methods are adopted, many systems remain limited to image-level detection and are not tightly coupled to the production-estimation objective.

In many agricultural contexts, isolated solutions focus on a single sub-problem: fruit detection, growth-stage recognition, segmentation, or network architecture evaluation. These solutions are valuable, but they do not necessarily resolve the broader challenge of orchard yield estimation. The gap is especially severe when a model counts fruit instances without linking them to fruit diameter, mass, visibility probability, or duplicate-removal across views. This is the fundamental reason why detection systems alone cannot be treated as complete production-estimation pipelines.

## 1.4 Role of Computer Vision and Machine Learning

Computer vision and machine learning have become central to orchard monitoring because they can process high-dimensional visual data, learn object patterns from annotated data, and operate at scale on images captured in situ. Detection models such as YOLO are especially effective for real-time localisation and object recognition. More advanced methods incorporate segmentation, geometry estimation, tracking, and temporal modelling to estimate developmental changes over time.

For pre-harvest production estimation, the value of these methods lies in their ability to connect visible fruit evidence to downstream variables such as diameter, mass, fruit maturity, and final yield. The repository’s figures show the breadth of this pipeline: YOLO-based fruit detection, tracking pipelines, segmentation masks and depth/3D projections, and economic trend visualisations that indicate how yield uncertainty interacts with decision-making. Together, these elements demonstrate the potential of vision-guided agricultural decision support.

## 1.5 Problem Statement

The problem addressed in this report is to develop a scientifically grounded and practically feasible methodological framework for pre-harvest pomegranate fruit detection, yield estimation, and production assessment using Python-based image processing, computer vision, and supporting economic interpretation. The framework must account for orchard complexity, canopy occlusion, multi-view duplication, fruit size variation, and the uncertainty in converting detected fruit into commercial production estimates.

In the context of this report, the main research challenge is not merely detection accuracy but the count-to-mass calibration problem: converting fruit counts and fruit dimensions into an estimate of orchard-level production that is meaningful for farm planning and economic decision-making.

## 1.6 Research Objectives

- To review and synthesise the literature on on-tree pomegranate phenotyping, fruit detection, and yield estimation from the relevant scientific studies and repository evidence.
- To evaluate the strengths and limitations of isolated detection, growth-stage, tracking, and estimation methods in the context of pre-harvest production forecasting.
- To identify the technical research gaps that prevent current methods from producing robust orchard-level production estimates.
- To propose a Python-based system architecture that integrates detection, segmentation, size estimation, and yield calibration.
- To present a validation protocol and evaluation framework suitable for orchard studies under uncertain field conditions.

## 1.7 Scope and Significance of the Study

The scope of the report is limited to the literature and repository materials relevant to pomegranate crop estimation and related production decision contexts. It includes detection, growth-stage classification, tracking, size and diameter estimation, and economic forecasting models that influence agricultural decisions. It does not claim to resolve all agronomic issues in orchard management, but it provides a structured foundation for a practical and research-driven yield-estimation pipeline.

The significance of the study lies in the fact that production estimation is both a technical problem and an agricultural decision problem. Accurate estimates improve labour allocation, harvest scheduling, storage planning, and supply forecasting. In the presence of market and price uncertainty, even moderately improved estimates can support more informed economic decisions over the crop cycle.

## 1.8 Organization of the Report

The remaining chapters follow the structure required for the academic report. The review begins with isolated phenotyping solutions and their limitations, continues with the temporal tracking dimension, and then introduces the economic context in which reliable production forecasts matter. This is followed by an explicit discussion of the research gaps, and finally by a proposed Python-based system architecture and evaluation protocol.

# 2. ON-TREE PHENOTYPING: ISOLATED SOLUTIONS AND LIMITATIONS

On-tree phenotyping methods attempt to estimate crop state directly from images of the tree canopy. Several papers in the review are directly relevant to pomegranate or fruit-tree crop sensing. Collectively, they reveal the fact that detection, size estimation, and development-stage classification are feasible but remain isolated solutions unless they are integrated into a yield-estimation framework.

## 2.1 Naseer et al. (2024): Pomegranate Growth-Stage Classification

**Full citation:** Naseer, A., et al. (2024). *Pomegranate Growth-Stage Classification*. [Repository literature context; retained with original paper title and author details where available in the final publication record.]

**Dataset & Hardware:** The study is aligned to image-based orchard phenotyping and is relevant to pomegranate crop development analysis in field conditions. The paper uses crop imagery acquired from orchards or controlled growth observation setups, with a focus on classification of growth stages rather than full orchard-level production measurement.

**Core Architecture/Method:** The method centres on image-based recognition of phenological stages and fruit development conditions. In general, such approaches rely on features derived from fruit appearance, canopy structure, and developmental morphology, often using deep learning or classical image descriptors for classification.

**Key Metrics:** The contribution is primarily methodological and classification-oriented. The key value of the paper is in identifying the progression of fruit development, which is necessary for any temporally grounded yield model.

**Critical analysis:** Growth-stage classification helps describe when fruit is developing and how canopy presentation changes over time. However, stage classification alone does not determine final mass or overall production. It is therefore an enabling step rather than a complete yield-estimation solution. It also remains sensitive to illumination, occlusion, and image acquisition geometry if the model is trained on narrow conditions.

## 2.2 Zhao et al. (2024): YOLO-Granada Fruit Detection

**Full citation:** Zhao, Y., et al. (2024). *YOLO-Granada: Fruit Detection and On-Tree Vision Pipeline*. [Use the exact paper bibliographic record in the final reference list.]

**Dataset & Hardware:** The publication is associated with orchard imaging and computer-vision-based fruit detection. The repository’s extracted figures indicate a model pipeline designed for fruit localisation and orchard-level visual recognition under real field conditions. This suggests acquisition using camera-based orchard imagery with a focus on efficiency and real-time detection capability.

**Core Architecture/Method:** The YOLO-Granada pipeline belongs to the family of single-stage object detectors and is designed to detect fruit within tree canopies. Its main strengths are fast inference, practical field deployment, and strong localisation performance for visible fruit. The architecture is well-suited to the early detection stage of a yield pipeline.

**Key Metrics:** The typical model outputs for this category include detection precision, recall, F1-score, and mean average precision. These metrics are relevant, although the repository does not provide exact numbers for every figure. The methodological significance is the direct detection of fruit objects from orchard images.

**Critical analysis:** YOLO-based detection substantially improves the speed and scalability of fruit recognition compared with manual inspection. However, a box or detection label is not equivalent to a count of unique fruit across the whole tree or orchard. Because fruit may be partially occluded, appear from multiple viewpoints, or be repeated across overlapping images, object detection alone does not guarantee a physically accurate production estimate. The method remains strongest as a front-end sensing component, not as a complete yield-estimation framework.

## 2.3 Devanna et al. (2025): On-Tree Detection and 3D Fruit-Size Estimation

**Full citation:** Devanna, F., et al. (2025). *On-Tree Detection and 3D Fruit-Size Estimation*. [Use the exact publication title and citation in the final references.]

**Dataset & Hardware:** The paper is associated with in-field sensing and machine-vision-based fruit measurement. The repository names and image dimensions suggest a robotic or image-based measurement system with segmentation, depth cues, and 3D fruit measurement. The hardware context likely includes camera rigs, depth sensing, and a structured measurement workflow.

**Core Architecture/Method:** The workflow includes detection, segmentation, and geometric estimation of fruit diameter or fruit size. This is critical because the physical mass of a fruit is not directly derivable from image detection alone. Several extracted figures show crop masks, depth/3D projection, and fruit-diameter estimation, indicating a pipeline that moves beyond object recognition and into physical measurement.

**Key Metrics:** The relevant evaluation indicators include fruit diameter estimation accuracy, segmentation quality, and correspondence between image geometry and physical fruit size. These measures matter because production estimation is sensitive to fruit mass and size distribution.

**Critical analysis:** This paper addresses a key weakness of detection-only approaches: image-based fruit size estimation. Even so, the challenge remains to translate fruit-size measurements into tree-level yield and orchard production given unknown fruit mass relationships, occlusion, and cross-view duplication. Therefore, 3D and size-estimation methods are necessary but not sufficient on their own. They need calibration with fruit mass and multi-view aggregation.

## 2.4 Comparative Analysis of On-Tree Phenotyping Methods

A comparative reading of the three phenotyping papers reveals three progressive steps in the technical pipeline: (i) recognition of crop stage, (ii) localisation of visible fruit, and (iii) measurement of physical fruit dimensions. The first step helps describe plant development, the second supports fruit counting and monitoring, and the third enables biological interpretation of fruit size or mass. The progression is important because it reflects an evolution from classification to sensing to measurement.

Nevertheless, the review also shows a recurring limitation: the reviewed methods provide useful isolated evidence but do not by themselves solve the agricultural production problem. Yield depends on both count and mass, and on the ability to aggregate fruit observations across the entire tree or orchard. The repository’s extracted figures and paper collection illustrate this gap clearly: the technical literature contains strong detection and measurement components, but their integration into a full production-estimation pipeline remains incomplete.

# 3. TEMPORAL TRACKING AND THE TIME DIMENSION

Crop yield is not static. Fruit development changes over time, and so does visibility, canopy overlap, and the probability of occlusion. This is why temporal tracking is essential: it allows fruit to be linked across observations, helps in harvest-date prediction, and reduces counting errors caused by repeated detections or missed detections over time.

## 3.1 Giménez-Gallego et al. (2025): On-Tree Tracking and Harvest-Date Prediction

**Full citation:** Giménez-Gallego, J., et al. (2025). *Fruit Monitoring and Harvest-Date Prediction Using On-Tree Automatic Image Tracking*. [Use the exact bibliographic version in the final reference section.]

**Dataset, Camera Setup, and Tracking Methodology:** This paper is explicitly concerned with on-tree automatic tracking and harvest-date prediction. The repository context indicates that the workflow uses time-series imagery of fruit on trees, with fruit association or tracking across successive observations. Such setups are inherently more informative than single-shot detection because they can follow the same fruit over time and interpret development trends.

**Core Architecture/Method:** The method combines object detection or segmentation with temporal association of fruit instances across image frames. The tracking module allows the system to estimate whether a detected fruit is the same object seen previously, to reduce duplicates and to monitor changes in size, maturity, or visibility. The prediction model then uses temporal descriptors to infer harvest date or crop development progression.

**Key Metrics:** The evaluation is usually framed in terms of tracking accuracy, association consistency, and temporal prediction quality. The main indicators include fruit association quality, prediction error, and time-series consistency. These metrics are more relevant to crop planning than one-off image classification.

**Critical analysis:** Time-series monitoring closes a major gap in single-image methods because it allows a crop to be interpreted as a dynamic phenomenon rather than a static object set. It also reduces gross miscounting in repeated observations. However, tracking models are still vulnerable to occlusion, leaf movement, and complex canopy structures. Additionally, a forecast of harvest date is not equivalent to a reliable estimate of orchard-level production unless it can be translated into counts, fruit size distributions, and mass calibration.

## 3.2 Dataset, Camera Setup, and Tracking Methodology

Time-series datasets are central to the temporal approach because the crop is observed across successive days or weeks. In orchard conditions, data collection commonly requires repeated image acquisition from fixed or semi-fixed camera placements, under varying illumination and canopy geometry. Typical setups carefully balance temporal resolution against the need for practical deployment. The point of repeated imaging is not only to see fruit but to maintain object identity across frames and estimate developmental transitions.

Tracking methodology can range from simple visual feature matching to more advanced association logic using motion, appearance, and spatiotemporal continuity. Without a robust tracking step, repeated detections may be treated as distinct fruits and the count becomes inflated. In yield-estimation tasks, temporal data are therefore a methodological necessity rather than an optional extension.

## 3.3 Temporal Feature Extraction and Prediction Results

Temporal feature extraction would typically include fruit appearance, change in visible diameter, maturity stage, shape consistency, and growth pattern over time. The key idea is that fruit development is not random; it follows a measurable progression that can be modelled. This makes the tracking and prediction stage highly relevant to pruning, harvest scheduling, and yield forecasting, especially in high-value fruit crops.

From the perspective of production estimation, temporal features are especially beneficial when they are linked to fruit sizing and local abundance. A model that recognises maturity progression is more useful than a model that only detects fruits at a single instant. Yet, this still requires calibration between visual growth and final fruit mass or commercial yield.

## 3.4 Critical Analysis and Limitations

Temporal tracking strengthens the scientific case for precision orchard monitoring, but it does not by itself resolve the core agricultural problem. Tracking can improve object identity consistency and enable harvest-date prediction, but orchard-level production still depends on a representative estimate of the number of fruit-bearing structures, their sizes, and their mass distribution. In practice, repeated observations can still miss fruit that remain hidden within dense canopies or under branch occlusion. Temporal methods therefore reduce but do not eliminate uncertainty.

## 3.5 Relevance to Pre-Harvest Production Estimation

Temporal tracking is directly relevant to the production-estimation problem because it enables a transition from static detections to dynamic crop-state estimation. It provides a way to evaluate the growth trajectory of fruit and the time at which fruit become economically mature. This is especially useful when the ultimate objective is to predict the harvest window or final crop output. The challenge remains to integrate growth tracking with robust counting and mass calibration so that the estimate can support actual orchard production assessment rather than mere fruit maturity prediction.

# 4. ECONOMIC CONTEXT AND DECISION UNCERTAINTY

Production estimation is not purely agronomic. Agricultural decisions are shaped by market dynamics, uncertainty, and economic value. A technically accurate fruit count may not be operationally useful unless it is connected to product quantity, price expectations, and orchard decision-making. This is why the economic literature is relevant to the report.

## 4.1 Mahajan et al. (2026): Agricultural Commodity Price Forecasting

**Full citation:** Mahajan, et al. (2026). *Agricultural Commodity Price Forecasting*. [Insert full bibliographic entry in the references section.]

**Dataset & Hardware:** The study is concerned with agricultural prices or market forecasting and is relevant to understanding how expected output influences market conditions. It represents the macro-level decision context within which fruit production estimates are useful.

**Core Architecture/Method:** Price forecasting models typically use historical trend analysis, time-series modelling, and structured economic predictors. These methods help interpret how expected supply, seasonal variability, and market instability affect final revenue.

**Key Metrics:** Forecast accuracy and error measures are central to economic modelling. Even if the exact metric values are not available in the repository, the role of the paper is conceptually clear: it demonstrates that agricultural decision-making depends on uncertainty-aware output forecasts.

**Critical analysis:** Agricultural profitability cannot be predicted from crop detection alone. Price forecasts and market models show why output uncertainty matters financially. If the estimate of supply is uncertain, then downstream planning becomes uncertain as well. The economic case for a reliable production model is therefore strong even when the technical challenge remains unresolved.

## 4.2 Nematollahi et al. (2025): Contract Farming Under Uncertainty

**Full citation:** Nematollahi, et al. (2025). *Contract Farming Under Uncertainty*. [Insert exact reference in the reference list.]

**Dataset & Hardware:** This work is relevant to decision-making in agricultural contracts, where the actual agricultural output is uncertain and the parties must manage risk under incomplete information.

**Core Architecture/Method:** Contract-farming models typically frame agreement design and coordination under uncertainty, where expected yield and price variability affect the terms of agricultural transactions.

**Key Metrics:** The relevant evaluation is often in economic terms such as expected utility, agreement value, and risk-adjusted decision outcomes rather than crop-science performance measures.

**Critical analysis:** This paper shows that uncertainty cannot be ignored in agricultural planning. A production-estimation system is therefore valuable not only because it counts fruits, but because it reduces uncertainty in planning and negotiation. Without reliable estimates, economic models are forced to work with broad assumptions and weak risk representation.

## 4.3 Abdurakhman et al. (2025): Fruit-Tree Lease Valuation

**Full citation:** Abdurakhman, et al. (2025). *Fruit-Tree Lease Valuation*. [Insert the exact bibliographic information.]

**Dataset & Hardware:** The study is oriented to agricultural asset valuation and long-term economic decisions. In orchard economics, tree condition, expected production, and future returns all influence the value of a lease or contractual arrangement.

**Core Architecture/Method:** Lease valuation models incorporate uncertainty in production, expected seasonal output, and the future value of the asset. These models are sensitive to the quality of crop-level forecasts and the level of confidence in production estimates.

**Key Metrics:** Economic decision metrics include valuation, expected returns, and risk-adjusted outcomes rather than technical detection accuracy. These are highly relevant to policy, leasing, and orchard investment decisions.

**Critical analysis:** The valuation literature reinforces the need for reliable pre-harvest information. If the yield estimate is weak, then agricultural asset valuation becomes speculative. This indicates that the technical challenge of crop estimation is not isolated from economics; it is connected to decisions that have direct financial consequences.

## 4.4 Comparative Analysis of Economic Models

Across price forecasting, contract farming, and lease valuation, a common theme emerges: uncertainty in agricultural output makes economic decision-making fragile. The economic models are potentially stronger when the underlying supply or yield estimate is better calibrated. In other words, economic decisions need better agronomic measurement at the production end of the pipeline. This makes the technical and economic parts of the problem complementary rather than separate.

## 4.5 Relevance of Reliable Production Estimates to Agricultural Decisions

Reliable production estimates matter because they reduce uncertainty in the decisions that farmers, traders, and investors must make. Such decisions include orchard labour planning, harvest timing, market positioning, contract negotiation, and asset valuation. In the pomegranate context, where fruit size and mass vary across production cycles, a production-estimation pipeline should therefore be designed not only for technical performance but also for practical decision utility.

# 5. RESEARCH GAPS: THE COUNT-TO-MASS CALIBRATION PROBLEM

Across the reviewed literature, a consistent research gap emerges: most methods offer detection or size estimation, but fewer provide a robust conversion from image-based fruit observations to orchard-level production and economic value. This is the central issue around which the report is framed.

## 5.1 Detection Versus Unique Fruit Counting

Object-detection models provide an estimate of visible fruit instances in an image. However, that is not necessarily the same as a unique fruit count for the tree or canopy. In orchard imagery, the same fruit may appear in multiple frames or from multiple viewpoints; in dense foliage it may also be partially or completely hidden. This makes a raw detection count susceptible to inflation or undercounting, and directly weakens the production estimate.

## 5.2 Canopy Occlusion and Visibility Limitations

Occlusion is one of the most significant unresolved issues in fruit detection. Leaves, branches, and other fruits can obscure the fruit surface so that detection is incomplete. The visible fruit distribution is therefore not the same as the true fruit distribution. Any production estimate that ignores detection probability under occlusion risks systematic bias.

## 5.3 Fruit Diameter Versus Physical Mass

Fruit diameter and size estimation are useful, but size is not equal to final mass or harvestable yield. Fruit mass depends on variety, maturity stage, water content, and fruit physiology. A model that estimates diameter without calibrating to fruit mass cannot directly estimate the commercial yield of a tree or orchard. This is one of the most important research gaps in the literature and is directly relevant to the objective of production assessment.

## 5.4 Multi-View Duplicate Detection and Temporal Tracking

When multiple images are captured from different angles or over time, the same fruit may be detected repeatedly. This problem can be made worse by view overlap, temporal drift, or differences in illumination. Multi-view fusion and temporal tracking are needed to disentangle duplicate counts from true fruit counts. Without this process, the estimate may reflect data acquisition structure rather than crop state.

## 5.5 Missing Link Between Fruit Sensing and Total Production

There remains a substantial gap between sensing fruit in a canopy and estimating total production for a tree or orchard. This missing link requires a multi-level model that connects detection, size, visibility probability, fruit mass calibration, and aggregation to tree-level and orchard-level output. The literature contains many useful sub-components but fewer integrated systems that connect them into a coherent production-estimation framework.

## 5.6 Limitations of Existing Economic Models for Physical Yield Estimation

Economic models are valuable for understanding uncertainty, pricing, and valuation, but they frequently rely on assumptions that do not explicitly incorporate physical orchard yield estimation. In other words, the agronomic production estimate and the economic model are often separated. This separation weakens the usefulness of agricultural decision support. A production model is more valuable when it is physically grounded and linked to market and valuation outcomes.

## 5.7 Comparative Research-Gap Table

| Research area | Main contribution | Primary limitation | Impact on production estimation |
|---|---|---|---|
| Growth-stage classification | Tracks phenological development | Does not determine final fruit mass or total yield | Provides contextual development information only |
| Fruit detection (YOLO-style) | Identifies visible fruit in orchard images | Counts visible objects rather than calibrated fruit load | Requires occlusion and duplicate correction |
| Fruit size estimation | Measures fruit diameter/geometry | Does not directly map to physical mass or total production | Needs calibration for mass and economic yield |
| Temporal tracking | Improves association across time | Can still fail under heavy canopy occlusion | Useful for time-series yield estimation but not sufficient alone |
| Economic forecasting and valuation | Models uncertainty and decision value | Often detached from physical orchard production estimation | Reflects the need for better agronomic supply estimates |

## 5.8 Final Research-Gap Statement

The central research gap can therefore be stated as follows: the literature provides strong evidence for fruit detection, stage recognition, temporal monitoring, size estimation, and economic decision modelling, but it does not yet provide a unified, robust, and calibrated method that transforms detections and size observations into reliable orchard-level pre-harvest production estimates. This is the count-to-mass calibration problem, and it is the core problem that motivates the proposed system architecture in the next chapter.

# 6. PROPOSED SYSTEM ARCHITECTURE AND EVALUATION PROTOCOL

## 6.1 Problem Statement and Core Objectives

A practical yield-estimation system should support the following tasks: fruit detection, canopy segmentation, object counting, fruit-size estimation, view fusion, temporal tracking, and tree-level/ orchard-level production inference. The output should be physically interpretable and suitable for agricultural decision-making rather than purely image-level classification.

The objectives of the proposed architecture are therefore to: (i) detect fruit reliably in the canopy, (ii) estimate geometric size and fruit condition, (iii) reduce duplicates and occlusion effects, (iv) relate visible fruit characteristics to mass or production, and (v) aggregate estimates to tree- and orchard-level production under uncertainty.

## 6.2 Proposed Python-Based System Architecture

The proposed system is designed as a modular Python pipeline. It begins with image acquisition and preprocessing, followed by fruit detection and segmentation, then size estimation and calibration, and finally aggregation and uncertainty analysis. A typical implementation would involve OpenCV for image processing, PyTorch or a detection framework for model inference, and analytical modules for mass calibration and yield aggregation.

The architecture can be organised into the following components:

- **Input module:** orchard images, camera metadata, temporal observations, and geolocation/tree identifiers.
- **Preprocessing:** colour correction, resizing, enhancement, and background-normalisation to reduce environmental variation.
- **Detection module:** fruit localisation and instance extraction using a YOLO-style or equivalent CNN pipeline.
- **Segmentation module:** fruit region isolation and boundary refinement for diameter estimation.
- **Geometry module:** conversion of segmented fruit to diameter, area, and 3D or projected size measures.
- **Calibration module:** mapping image-derived fruit dimensions to physical mass or yield per fruit.
- **Tracking/fusion module:** duplicate removal across views and temporal observations.
- **Aggregation module:** tree-level and orchard-level yield estimation from calibrated fruit measurements.
- **Uncertainty module:** confidence bounds, sensitivity analysis, and decision support under occlusion and measurement variability.

## 6.3 System Workflow and Module Description

At a workflow level, the system begins by capturing a time series of orchard images. These images are preprocessed and then passed to the detection model to identify fruit candidates. Candidate instances are segmented, measured, and associated across image views or time points. The yield estimator then aggregates fruit-level values into tree-level estimates and orchard-level production estimates by applying a calibration function that relates visible fruit dimensions to crop mass or yield.

The report emphasises that processing should remain modular. Each stage can be evaluated independently while the final estimate is produced only after successful fusion of detection, geometry, and calibration outputs. This is scientifically important because isolated module performance does not guarantee end-to-end yield accuracy.

## 6.4 Dataset Requirements and Image Acquisition

The dataset should include images sampled across growth stages, multiple trees, varying lighting, and repeated capture over time. It should include annotations of fruit positions, segmentation masks, dimensions, and associated metadata such as date, tree identifier, camera viewpoint, and season. The acquisition pipeline must be designed to support both detection and calibration tasks, not only classification.

A robust dataset would support model development in three stages: training, validation, and stress testing. Stress tests should include heavy occlusion, partially visible fruits, low-light conditions, and canopy-density variability. Without these conditions, the model may appear accurate in controlled conditions but fail in operational orchard environments.

## 6.5 Fruit Detection, Segmentation, and Counting

Fruit detection is performed on orchard images using a YOLO-style detector or equivalent high-performance deep-learning framework. The detected instances are then segmented to separate fruit from leaves, branches, and the background. Segmentation improves both the reliability of the count and the accuracy of the size estimate because it allows the fruit boundary to be measured rather than approximated from a bounding box alone.

The count module should be designed to distinguish between visible fruit count, unique fruit count, and estimated total fruit load. Visible fruit count is the number of fruit instances in an image; unique fruit count is the number of distinct fruit observed across views or frames; estimated total fruit load is the orchard-level estimate after calibration and inference under occlusion. The distinction is important because production estimation is fundamentally about the latter, not the former.

## 6.6 Fruit-Size Estimation and Mass Calibration

Fruit-size estimation can be based on several geometric outputs, including projected diameter, contour length, area, or 3D depth-derived dimension. These measurements are then mapped to fruit mass using a calibration model. This is the critical stage at which the system moves from detection to production estimation. Without a mass curve or calibration relation, size measurements are not sufficient for yield assessment.

The calibration equation can be expressed in a general form as:

m_i = f(d_i, g_i, s_i, θ_i)

where m_i is the fruit mass, d_i is the image-derived size or diameter, g_i denotes fruit geometry, s_i represents maturity or growth stage, and θ_i denotes model or orchard-specific calibration parameters.

## 6.7 Tree-Level Yield and Orchard Production Estimation

At the tree level, the estimate is formed by summing the calibrated mass for all detected and inferred fruit according to the model. For a tree t, the production estimate can be represented generically as:

Y_t = Σ_{i=1}^{N_t} m_i × ρ_i

where Y_t is the estimated yield, N_t is the estimated fruit count for the tree, and ρ_i represents a correction factor for visibility, duplicate removal, or fruit-specific weight adjustment. The orchard-level estimate is then obtained by aggregation across trees:

Y_O = Σ_{t=1}^{T} Y_t

This formulation clarifies that orchard-level production estimation depends on the connection between individual fruit measurements and tree-level aggregation. The system should therefore support both fruit-level and aggregate-level reporting.

## 6.8 Mathematical Formulation and Assumptions

The pipeline should be built on a set of practical assumptions. These include: (a) fruit instances are detectable using a vision model under the given canopy conditions, (b) image-derived size is correlated with physical mass, (c) duplicate detections can be suppressed through tracking or multiview fusion, and (d) the calibration relation is stable within a crop season or a defined orchard environment. These assumptions are reasonable in orchard monitoring but must be validated carefully rather than assumed automatically.

The final estimate is therefore not a single deterministic value but a probability-informed estimate subject to uncertainty. In engineering terms, the system should report confidence intervals or uncertainty bands along with the estimated yield. This is particularly important when the production estimate is used in contract decisions or orchard planning.

## 6.9 Validation Protocol and Evaluation Metrics

| Component | Evaluation objective | Recommended metric |
|---|---|---|
| Fruit detection | Ability to localise fruit in orchard images | Precision, Recall, F1-score, mAP |
| Segmentation | Boundary quality and fruit region separation | IoU, Dice score, boundary error |
| Fruit size estimation | Correlation between estimated geometric size and physical size | MAE, RMSE, R², percentage error |
| Counting | Counter accuracy after duplicate correction and occlusion handling | Counting error, absolute error, relative error |
| Yield estimation | Consistency of tree/orchard production estimate | Bias, RMSE, calibration error, uncertainty coverage |

Validation should proceed in stages: fruit detection, segmentation, size estimation, tracking, and finally total production estimation. A system that performs well in the first stage but fails in the final aggregation stage should not be considered successful for agricultural decision-making.

## 6.10 Expected Outcomes

The expected outcomes of the proposed system are a scientifically grounded method for pre-harvest yield estimation that can be implemented in Python and adapted to orchard conditions. The method should provide useful estimates of fruit count, fruit size, and tree- or orchard-level yield while making uncertainty explicit. This would provide a stronger basis for decision-making than detection-only approaches.

## 6.11 Scope Boundaries, Feasibility, and Implementation Roadmap

The proposed method is feasible for a project or thesis context because the underlying components are already represented in the literature and in the extracted repository figures. The challenge is the integration of components into a single calibrated pipeline rather than the invention of entirely new technology. In practical terms, the system can be implemented in iterative stages: image collection, object detection, size estimation, fruit tracking, mass calibration, and aggregation.

The realistic roadmap is therefore incremental. An initial implementation may focus on the single-tree case with controlled imaging. The next step is to extend the model to multiple trees and field conditions. The final stage is orchard-wide production estimation with uncertainty quantification and economic interpretation.

## 6.12 Limitations and Future Work

Despite the promise of the proposed framework, several limitations remain. Detection will continue to be affected by occlusion and variable illumination. Size estimation may not generalise across cultivars or orchard conditions without calibration. Temporal tracking may fail when fruit motion or view geometry is unstable. And the mass calibration problem remains a central challenge because physical yield depends on fruit attributes that are not captured by image patterns alone.

Future work should therefore focus on calibration datasets linking fruit geometry to physical mass, cross-view and temporal fusion methods, and uncertainty-aware reporting. In addition, there is a need to connect crop production estimates with economic decision models so that the output is actionable for orchard managers and agricultural stakeholders.

# REFERENCES

1. Abdurakhman, A., et al. (2025). *Fruit-Tree Lease Valuation*. [Insert the correct full bibliographic reference in the final formatted bibliography.]
2. Devanna, F., et al. (2025). *On-Tree Detection and 3D Fruit-Size Estimation*. [Insert full publication details.]
3. Giménez-Gallego, J., et al. (2025). *Fruit Monitoring and Harvest-Date Prediction Using On-Tree Automatic Image Tracking*. [Insert final bibliographic entry.]
4. Mahajan, et al. (2026). *Agricultural Commodity Price Forecasting*. [Insert final bibliographic entry.]
5. Naseer, A., et al. (2024). *Pomegranate Growth-Stage Classification*. [Insert full bibliographic entry.]
6. Nematollahi, A., et al. (2025). *Contract Farming Under Uncertainty*. [Insert full bibliographic entry.]
7. Zhao, Y., et al. (2024). *YOLO-Granada: Fruit Detection and On-Tree Vision Pipeline*. [Insert full bibliographic entry.]

Note: In the final polished document, the full reference list should be standardised to a consistent academic format (APA, Harvard, or IEEE) and should include complete author names, publication venue, year, DOI or identifier, and page details where available.

## Appendix A. Repository Figure Notes

The repository contains extracted figures that support the structure of this report. These figure assets are valuable for presentation and academic narration, and include YOLO-based detection outputs, temporal tracking figures, size-estimation diagrams, and economic charts. The relevant file categories are:

- YOLO-Granada / fruit detection and growth-stage figures
- Fruit_Tracking_Harvest_Prediction / temporal tracking and prediction figures
- Pomegranate_Size_Estimation_Robot / segmentation, depth, 3D projection, and fruit diameter estimation
- Agricultural_Price_Forecasting / market and trend charts
- Fruit_Tree_Lease_Valuation / valuation model figures
- Agricultural_Production_Research / yield and production figures

These figures should be retained with source attribution in the final polished academic version. Their role is to reinforce the conceptual narrative and to show how the literature converges on the central count-to-mass and production-estimation challenge.

# Final Note

This draft has been organised according to the exact structural requirements supplied by the user and grounded in the repository evidence. It intentionally avoids inventing numerical results or unsupported claims. The final publication-ready version should replace the bracketed placeholder bibliographic entries with the exact paper titles, authors, venues, DOIs, and metrics as they appear in the original sources.

