# Extending molecular scaffolds with building blocks

MoLeR is a graph-based generative model that combines fragment-based and atom-by-atom generation of new molecules with scaffold-constrained optimization. It does not depend on generation history and therefore MoLeR is able to complete arbitrary scaffolds. The model has been trained on the GuacaMol dataset. Here we sample the 300k building blocks library from Enamine.

This model was incorporated on 2023-11-03.


## Information
### Identifiers
- **Ersilia Identifier:** `eos633t`
- **Slug:** `moler-enamine-blocks`

### Domain
- **Task:** `Sampling`
- **Subtask:** `Generation`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `Chemical graph model`, `Compound generation`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `100`
- **Output Consistency:** `Variable`
- **Interpretation:** 1000 new molecules are sampled for each input molecule, preserving its scaffold.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| smiles_000 | string |  | Generated compound index 0 using MoLeR and Enamine building blocks |
| smiles_001 | string |  | Generated compound index 1 using MoLeR and Enamine building blocks |
| smiles_002 | string |  | Generated compound index 2 using MoLeR and Enamine building blocks |
| smiles_003 | string |  | Generated compound index 3 using MoLeR and Enamine building blocks |
| smiles_004 | string |  | Generated compound index 4 using MoLeR and Enamine building blocks |
| smiles_005 | string |  | Generated compound index 5 using MoLeR and Enamine building blocks |
| smiles_006 | string |  | Generated compound index 6 using MoLeR and Enamine building blocks |
| smiles_007 | string |  | Generated compound index 7 using MoLeR and Enamine building blocks |
| smiles_008 | string |  | Generated compound index 8 using MoLeR and Enamine building blocks |
| smiles_009 | string |  | Generated compound index 9 using MoLeR and Enamine building blocks |

_10 of 1000 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos633t](https://hub.docker.com/r/ersiliaos/eos633t)
- **Docker Architecture:** `AMD64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos633t.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos633t.zip)

### Resource Consumption
- **Model Size (Mb):** `32`
- **Environment Size (Mb):** `2086`


### References
- **Source Code**: [https://github.com/microsoft/molecule-generation](https://github.com/microsoft/molecule-generation)
- **Publication**: [https://arxiv.org/abs/2103.03864](https://arxiv.org/abs/2103.03864)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2021`
- **Ersilia Contributor:** [miquelduranfrigola](https://github.com/miquelduranfrigola)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [MIT](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos633t
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos633t
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
