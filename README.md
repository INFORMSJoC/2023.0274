[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Social Network Prediction Problems: Using Meta-Paths and Dynamic Heterogeneous Graph Representation for Label Propagation

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](https://github.com/negarmaleki96/2023.0274/blob/ef429ace556643abbca1f97dd1f4758d4ddbae97/LICENSE.txt).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[Social Network Prediction Problems: Using Meta-Paths and Dynamic Heterogeneous Graph Representation for Label Propagation](https://doi.org/10.1287/ijoc.2023.0274) by N. Maleki, B. Padmanbhan, and K. Dutta. 
The snapshot is based on 
[this repository](https://github.com/negarmaleki96/LabelPropagation_MetaPath) 
in the development repository. 

**Important: This code is being developed on an on-going basis at 
https://github.com/negarmaleki96/LabelPropagation_MetaPath. Please go there if you would like to
get a more recent version or would like support**

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2023.0274

https://doi.org/10.1287/ijoc.2023.0274.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{Social Network Prediction Problems: Using Meta-Paths and Dynamic Heterogeneous Graph Representation for Label Propagation,
  author =        {N. Maleki, B. Padmanbhan, and K. Dutta},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Social Network Prediction Problems: Using Meta-Paths and Dynamic Heterogeneous Graph Representation for Label Propagation}},
  year =          {2025},
  doi =           {10.1287/ijoc.2023.0274.cd},
  url =           {https://github.com/INFORMSJoC/2023.0274},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0274},
}  
```

## Description

This work introduces a dynamic heterogeneous graph representation that integrates time into both nodes and edges, enabling a more accurate modeling of multiplex and evolving relationships in social networks. We further propose **Meta-paths + LPA**, an extension of the Label Propagation Algorithm that incorporates temporal meta-paths for improved classification on dynamic heterogeneous structures. The framework is demonstrated through a case study on **Steemit**, showcasing its ability to handle complex time-dependent queries and prediction tasks.

## Results

We evaluate our proposed **Meta-paths + LPA** approach on three benchmark datasets, comparing it against four state-of-the-art algorithms for the category prediction task. The results show that our method consistently outperforms existing approaches, achieving at least **13.79% higher accuracy** across datasets.  

Key findings:  
- **Superior Accuracy**: Meta-paths + LPA achieves significant performance improvements over baseline methods.  
- **Temporal Advantage**: Incorporating time into both nodes and edges allows the model to capture evolving relationships that conventional graph representations miss.  
- **Generalizability**: The proposed framework demonstrates effectiveness across multiple datasets and domains, highlighting its robustness.  
- **Real-World Validation**: The Steemit case study illustrates the practicality of the method for dynamic querying and predictive tasks in real social networks.

## Replicating

To replicate the results, do 

```
python creating_graph.py
python nx_to_PyG.py
python metapath_LP.py
```

## Support

For support in using this software, submit an
[issue](https://github.com/negarmaleki96/LabelPropagation_MetaPath/issues).
