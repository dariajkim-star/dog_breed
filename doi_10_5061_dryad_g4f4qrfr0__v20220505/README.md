## Manuscript
### Associated data files and scripts for 'Ancestry-inclusive dog genomics challenges popular breed stereotypes'

The following archives contain data files and scripts associated with the manuscript:

> K. Morrill, J. Hekman, X. Li, J. McClure, B. Logan, L. Goodman, M. Gao, Y. Dong, M. Alonso, E. Carmichael, N. Snyder-Mackler, J. Alonso, H. J. Noh, J. Johnson, M. Koltookian, C. Lieu, K. Megquier, R. Swofford, J. Turner-Maier, M. E. White, Z. Weng, A. Colubri, D. P. Genereux, K. A. Lord, E. K. Karlsson, Ancestry-inclusive dog genomics challenges popular breed stereotypes. *Science*. 2022.

---

## Data Files

### Genetic Data:

See `GeneticData.zip` for the archive of genetic data.

#### Darwin's Ark Cohort

A *PLINK1* bfile set (.bed/.bim/.fam) with sample IDs encoded as dog IDs matching individuals from the Darwin's Ark survey data files, including genotypes from 1,715 dog whole genomes sequenced under BioProject PRJNA675863.

`DarwinsArk_gp-0.70_snps-only_maf-0.02_geno-0.20_hwe-midp-1e-20_het-0.25-1.00_N-2155.bed`

`DarwinsArk_gp-0.70_snps-only_maf-0.02_geno-0.20_hwe-midp-1e-20_het-0.25-1.00_N-2155.bim`

`DarwinsArk_gp-0.70_snps-only_maf-0.02_geno-0.20_hwe-midp-1e-20_het-0.25-1.00_N-2155.fam`

See Supplementary Materials from manuscript for methods and more details.

#### Mendel's Mutts Cohort

A variant call file containg variants and genotype records for the 27 mixed-breed dog whole genomes sequenced under BioProject PRJNA683923.

`MendelsMutts_WGS_BioProject-PRJNA683923_variant-only.vcf.gz`

`MendelsMutts_WGS_BioProject-PRJNA683923_variant-only.vcf.gz.tbi`

See Supplementary Materials from manuscript for methods and more details.

---

### Reference Data:

See `ReferenceData.zip` for the archive of genetic data.

#### `ReferenceData_breeds.csv`
Table of breed names and pseudonyms to link across data sets.

    "breed_name"
    Breed name in Darwin's Ark survey data.
    
    "breed_name_short"
    Shortened breed name.
    
    "abbr"
	 Abbreviated breed name.
	 
    "umbrella_name"
    Grouped breed name.
    
    "breed_calling"
    Breed name in Darwin's Ark breed ancestry results.
    
    "breed_muttmix"
    Breed name in Mutt Mix survey data.

#### `ReferenceData_stereotypes_scores.csv`
Breed temperament scores from the American Kennel Club.

#### `ReferenceData_stereotypes_groups.csv`
Breed groups and three word descriptors from the American Kennel Club.

#### `ReferenceData_AKC_registrations.csv`
Breed registrations to the American Kennel Club averaged over 2000 to 2015.

#### `ReferenceData_breed_stature.csv`
Breed average height-to-withers in centimeters.

#### `ReferenceData_breed_standards_byQuestion.csv`
Breed standard values by Darwin's Ark survey question.

---

### MuttMix Data:

See `MuttMix.zip` for the archive of MuttMix data.

#### `MuttMix_20180616_survey_data.csv`
The raw results for the MuttMix survey which closed data collection on June 16th, 2018.

    "UserID"
    Registration ID of participant.
    
    "DateAdded"
    Date on which participant recorded guess.
    
    "ProfDogTrainer"
    Whether participant has professional experience with dogs (1) or not (0).
    
    "DogFirstName"
    Call name of dog.
    
    "DogID"
    Survey ID of dog.
    
    "BreedChoice1"
    Guess for 1st top breed.
    
    "BreedChoice2"
    Guess for 2nd top breed.
    
    "BreedChoice3"
    Guess for 3rd top breed (optional).

#### `MuttMix_20180616_genetic_data.csv`
The genetically inferred breed ancestry results for dogs in the MuttMix survey.

    "dog"
    Survey ID of dog.
    
    "breed"
    Breed from which ancestry inferred.
    
    "pct"
    Percent global ancestry.
    
    "survey_option"
    Breed from which ancestry inferred, if option in the MuttMix survey (which isn't all breeds).
    
#### `MuttMix_20180616_trait_data.csv`
The physical traits of dogs in the MuttMix survey.

#### `MuttMix_20180616_pheno_entropy.csv`
Entropy in breed guesses explained by physical traits of N-1 dogs in the MuttMix survey.

---

### Mendel's Mutts Data:

See `MendelsMutts.zip` for the archive of data associated with the Mendel's Mutts cohort.

#### `MendelsMutts_breedcalls.csv`
Genetically-inferred ancestry in the Mendel's Mutts cohort.

#### `MendelsMutts_runs-of-homozygosity.csv`
Runs of homozygosity detected in the Mendel's Mutts cohort as well as the Canine Diversity VCF.

#### `MendelsMutts_sequencing-data_sample-info.csv`
Sample information for dogs with whole genome sequencing data.

#### `MendelsMutts_extent-of-LD.csv`
Input data for the LD analysis.

#### `MendelsMutts_variants-tagged.csv`
Input data for variant tagging analysis.

---

### Darwin's Ark Data:

See `DarwinsArk.zip` for the archive of data associated with the Darwin's Ark cohort.

#### `DarwinsArk_20191115_dogs.csv`
Information for dogs enrolled in Darwin's Ark on or before freeze date of November 15th, 2019.

    "id"
    The dog's ID in Darwin's Ark database.

    "sex"
    The owner-assigned sex on dog profile.

    "sterilized"
    The owner-assigned spay/neuter status on dog profile.

    "birth_date"
    The standardized birth date of the dog from owner-provided information.

    Prior to July 3rd, 2018, age and birth dates for enrollment were collected as free response entries. In order to standardize these as birth dates in international format (YYYY-MM-DD) for estimation of age, the following steps were executed using a combination of functions from the R packages data.table, stringr, anytime, and lubridate:
    - For dogs with a parsable birth date, directly convert to YYYY-MM-DD.
    - For dogs with a year and month, assign birth date YYYY-MM-01.
    - For dogs with a year only, assign birth date YYYY-01-01.
    - For dogs with no parsable birth date but age given in years and/or months, parse into duration and subtract from date of earliest survey response to estimate birth date.

    After each of the above steps:
    - Remove all birth dates before January 1st, 1980
    - Remove all birth dates after data freeze date (November 15th, 2019)
    The remainder with age or birthday entries were parsed by hand, if possible.

    "flagged_deceased_date"
    The date that the owner flagged their dog as deceased on the dog's profile.

    "region"
    Region in the United States of America based on zip code information, if owner provided an address in the United States of America.

    "environ"
    Urban, suburban, or rural environ given population density (from US Decennial Census of 2010) based on zip code information, if owner provided an address in the United States of America.

    "origin"
    Owner response to question 117, "Where did you get DOG?"

    "size"
    Owner response to question 121, "When DOG is standing next to someone of average height, how high are HIS shoulders?"

    "breed1"
    Primary owner-reported known or suspected breed.

    "breed2"
    Secondary owner-reported known or suspected breed.

    "purebred"
    Owner input for whether dog has purebred registration.

    "owner_breed"
    Breed assigned for any dog with a single owner-reported breed (breed1) and no other breed.

    "reg_breed"
    Breed assigned for any dog with a single owner-reported breed (breed1) and having registered purebred status (purebred == "yes").

    "regseq_breed"
    Breed assigned for any dog with reg_breed or with at least 85% of a breed inferred in their global breed ancestry.

    "geno_wgs"
    Dog has high-coverage whole genome sequencing data.

    "geno_axiom"
    Dog has genotyping array data from the Axiom Canine Genotyping Array Set A & B.

    "geno_lowpass"
    Dog has low-coverage whole genome sequencing data from the Gencove platform.

    "other_population"
    Whether breed reported by owner is a landrace or village dog, named crossbreed, or wild canid.

    "owner_label"
    Non-breed population label reported by owner.
    
    "responses"
    Number of survey responses observed in answers table.
    
    "response_rate"
    Proportion of questions with responses observed in answers table.
    
    "mutt"
    Final classification as a mutt.
    
    "cand"
    Final classification as a candidate purebred dog.
    
    "conf"
    Final classification as a confirmed purebred dog.
    
    "consensus_breed"
    Final breed assignment.

#### `DarwinsArk_20191115_questions.csv`
Survey questions.

#### `DarwinsArk_20191115_answers.csv`
Survey responses.

#### `DarwinsArk_answers_forFactorAnalysis.txt`
Survey responses used for factor analysis.

#### `DarwinsArk_20191115_factors.csv`
Factors discovered through exploratory factor analysis.

#### `DarwinsArk_20191115_factor_scores.csv`
Factor scores for dogs with sufficient survey responses.

#### `DarwinsArk_20191115_heritability.csv`
Genome-wide complex trait analysis (GCTA) restricted maximum likelihood (REML) SNP-based heritability estimates using genetic relationship matrices (GRM) calculated (1) from all SNPs, (2) from SNPs stratified by LD score, or (3) as partitioned by SNPs within/without LD-based clumping of genome-wide associations (*p* < 1e-6), as well as additional heritability analysis of unrelated (kinship < 0.2) and highly admixed (no breed ancestry >45%) dogs with the top 10 principal components included in the models.

#### `DarwinsArk_20191115_inbreeding.csv`
Coefficients of inbreeding estimated from runs of homozygosity.

	dog - Dog ID
	F.roh - ROH-estimated COI
	n.seg - number of runs of homozygosity
	kb.total - total length of runs
	kb.avg - average length of runs

#### `DarwinsArk_20191115_kinship.csv`
Kinship between dogs as measured by KING-robust kinship estimator.

#### `DarwinsArk_20191115_breedcalls.csv`
Results of supervised admixture analysis giving inferred global breed ancestry calls.

#### `DarwinsArk_20191115_K-75_unsupervised_clusters.csv`
Results of unsupervised admixture analysis giving inferred cluster fractions for K=75 clusters estimated from chromosomes 1, 15, and 38. Note that cluster IDs are unique to the chromosome on which unsupervised admixture analysis was performed.

#### `DarwinsArk_20191115_relative_risk.csv`
Results of relative risk analysis.

#### `DarwinsArk_20191115_website.descriptions.csv`
Input for interactive dashboard.

#### `DarwinsArk_20191115_website.factors.csv.gz`
Input for interactive dashboard.

#### `DarwinsArk_20191115_website.questions.csv.gz`
Input for interactive dashboard.

#### `DarwinsArk20191115_anova.csv`
Results of the ANOVA analyses.

#### `DarwinsArk_20191115_survey_permutations.csv`
Results of the survey permutation-based population peculiarity score analyses.

#### `DarwinsArk_20191115_LMER_models.csv`
Results of the linear mixed effects regression models of breed ancestry for question and factor scores in highly admixed dogs with no breed ancestry >45%.

#### `DarwinsArk_20191115_magma.csv`
Results of MAGMA gene set enrichment on genome-wide association summary statistics.

#### `DarwinsArk_20191115_GWASloci_breedPBS_permutations.csv`
Breed differentiation per association locus relative to randomly permuted loci.

---

## Script Files

#### Basic Statistics

The *R* script `DarwinsArk_basic-statistics.R` in the `Scripts_Analysis.zip` archive contains commands to generate several plots and summary statistics included in the manuscript.

#### Cumulative Distribution of Variant Discovery

The *R* script `variant_discovery_cumulative_distribution.R` in the `Scripts_Analysis.zip ` archive contains commands used for the *Cumulative variant discovery* analysis.

See the Supplementary Materials of the manuscript for methods.

#### Population Peculiarity Scores (PPS)

The archive `Scripts_PopulationPeculiarityScoring.zip` contains scripts for the *Population Peculiarity Scoring (PPS)* analysis. See `README` in archive for instructions on how to run the population peculiarity score analysis. Further details are commented within the *R* scripts.

Also see the Supplementary Materials of the manuscript for methods.

#### Linear Mixed-effects Regressions (LMER) Models

The *R* script `linear_mixed_effects_regressions.R` in the `Scripts_Analysis.zip ` archive contains commands used for the *Linear mixed-effects regression models (LMERs)* analysis.

See the Supplementary Materials of the manuscript for methods.

#### Plotting Scripts

A series of plotting scripts are given in the archive `Scripts_Plotting.zip`.

#### Dashboard Scripts

Scripts used to generate data for the interactive dashboard are contained in the archive `Scripts_Dashboard.zip`.

---

## Contact

If you have further questions about the data and script file archive, then contact corresponding authors Kathleen Morrill (kathleen.morrill@umassmed.edu) and Elinor Karlsson (elinor.karlsson@umassmed.edu), and we will resolve them as soon as possible.