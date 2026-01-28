# MultiModX

**Strategic and Tactical Evaluation of Air-Rail Multimodal Networks** (SESAR Solution 399) from the MultiModX project.


| **MultiModX**  | **Project Info**                                                                                                                                                                                                                                                 |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Title**        | Integrated Passenger-Centric Planning of Multimodal Transport Networks: **MultiModX**                                                                                                                                                                            |
| **DOI / CORDIS** | [10.3030/101114815](https://doi.org/10.3030/101114815)                                                                                                                                                                                                           |
| **Duration**     | June 2023 - December 2025                                                                                                                                                                                                                                        |
| **Partners**     | - The University of Westminster (UoW) <br> - Nommon Solutions and Technologies (Nommon) <br> - Technische Universitaet Dresden (TUD)<br> - Bauhaus Luftfarhrt (Bauhaus) <br> - Union Internationale Des Chemins de fer (UIC)<br> - Airport Regions Council (ARC) |
| **Website**      | [https://multimodx.eu/](https://multimodx.eu/)                                                                                                                                                                                                                   | 
  


## Key information

This repository contains the code of the **Strategic and Tactical Evaluation of Air-Rail Multimodal Networks**. This is 
SESAR **Solution 399** coordinated by the University of Westminster 
([Centre for Air Traffic Management Research](https://blog.westminster.ac.uk/atm-team/)) within the MultiModX project.


Solution 399 – Multimodal Performance Evaluation measures door-to-door passenger outcomes using a structured catalogue 
of passenger-centric indicators, extending existing aviation performance frameworks to multimodal journeys and 
providing open-source tools for evaluating multimodal networks.

### Solution 399 Components

Solution 399 is composed of four main elements:
1. A **Multimodal and Passengers Experience Performance Framework**: A [Digital Catalogue of Indicators](https://nommon.atlassian.net/wiki/external/MzA2ZTJmMjU5MDUyNDNlYzlkNDBmNTMwOTRlMDY4MGY) that
complements and extends the SESAR Performance Framework to include passenger-centric
aspects capturing door-to-door multimodal itineraries.
2. A **Strategic Multimodal Evaluator**: This evaluator can assess planned air, rail and multimodal networks. From the 
demand and supply (flight and rail schedules and infrastructure aspects), the  model computes the realisation of the 
network.
3. A **Pre-Tactical Multimodal Evaluator**: This is a functionality which can evaluate the impact on passengers of the 
replanning of air and rail operations, e.g. in the case of disruptions.
4. A **Tactical Multimodal Evaluator**: Simulates the day of operations with a focus on the air-rail connectivity using
an extension of [Mercury](https://github.com/UoW-ATM/Mercury/tree/multimodx).

The models are complemented by:
1. A **Performance Indicators Computation**: Functionalities to compute PI and KPIs from the outcome of the Evaluators.
2. A **Strategic Dashboard**: To visualise some of the Strategic Evaluator results.


## Documentation

- **Technical documentation** in [https://uow-atm.github.io/MultiModX](https://uow-atm.github.io/MultiModX/).


### Key Publications

- **White Paper** from MultiModX project final results: [MultiModX_White_Paper.pdf](docs/publications/documents/MultiModX_White_Paper.pdf)
- **Technical Summary** of Multimodal Performance Framework and Multimodal Evaluators:
[Technical_Summary_Open_Multimodal_Performance_Framework_and_Evaluation_tools.pdf](docs/publications/documents/Technical_Summary_Open_Multimodal_Performance_Framework_and_Evaluation_tools.pdf)


### Articles 

  - Delgado, L., Weiszer, M., de Boissieu, M., Bueno-Gonzlez, J. and Menendez-Pidal, L. (2025). [Strategic multimodal 
    evaluator for air-rail networks](docs/publications/articles/EWGT_2025_MMX_Strategic.pdf). 27th Euro Working Group on Transportation Meeting (EWGT 2025).
  - Weiszer, M., Delgado, L. and Menendez-Pidal, L. (2025). [Air-rail multimodal disruption management - Rail network 
    supporting air disruptions](docs/publications/articles/SIDs_2025_paper_108-final.pdf). SESAR Innovation Days 2025.
  - Weiszer, M., Delgado, L. and Gurtner, G. (2024).
    [Evaluation of Passenger Connections in Air-rail Multimodal Operations](docs/publications/articles/SIDs_2024_paper_050_final.pdf). 
    SESAR Innovation Days 2024.


## License

MultiModX evaluators are released under the GPL v3 licence. The licence can be found
in [LICENSE](LICENSE).
