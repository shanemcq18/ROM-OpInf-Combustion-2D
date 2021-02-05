#!/bin/bash
# THIS IS A HISTORY OF COMMANDS TO GET THE FIGURES IN THE REPORT.
# DO NOT RUN THIS FILE (MANY DEPRECATIONS).

# Unpack raw, unzipped data.
python3 step1.py

# Produce lifted, scaled snapshots sets with 10000, 15000, and 20000 training points.
python3 step2a_lift.py 10000 15000 20000

# For each training set,
for trainsize in 10000 15000 20000; do
    # Get a POD basis of rank 60.
    python3 step2b_basis.py ${trainsize} 60

    # Make projected data sets of dimension 17-50.
    python3 step2c_project.py ${trainsize} 17 50 --moderange
    
    # Learn ROMs of dimension 17-50 with various regularizations.
    python3 step3_train.py ${trainsize} --modes 17 50 --moderange
    
    # Make all of the interesting visualizations.
    ## Check to see which models are stable over the whole time domain.
    python3 step4_analysis.py ${trainsize} --modes 17 50 --moderange --test-stability
    ## L-curves for each of the trained models.
    python3 step4_analysis.py ${trainsize} --L-curve --modes 17 50 --moderange
    ## Stability test for each trained model.
    python3 step4_analysis.py ${trainsize} --test-stability --modes 17 50 --moderange
    ## Basis size plots for a few regularization values at several time locations.
    python3 step4_analysis.py ${trainsize} --basis-vs-error -reg 6e4 -snap 10000 15000 20000 25000 30000
    ##
done

# Extra large model for Tecplot animations.
python3 step2b_basis.py 20000 100
python3 step2c_project.py 20000 94
python3 step3_train.py 20000 -r 94 -reg 1e4

# Choose the best regularization for each chosen model.
python3 step4_analysis.py  5000 --best-regularization --modes 12 17 24
python3 step4_analysis.py 10000 --best-regularization --modes 23 33 48
python3 step4_analysis.py 20000 --best-regularization --modes 44 62 94

# Specific plots for the slides.
python3 step4_analysis.py 10000 --basis-vs-error --modes 17 60 --moderange -reg 6e05 7e04 9e04 1e05 6e04 7e04 7e04 9e04 9e04 1e05 4e05 2e05 4e04 6e04 4e04 6e04 4e04 4e04 2e04 4e04 2e04 8e04 8e04 6e04 2e05 3e04 3e04 3e04 3e05 3e04 2e04 3e04 4e04 3e04 2e04 3e04 2e04 3e04 1e04 3e04 1e04 2e04 1e04 2e04 --timeindex 10000 10500 20000 30000
python3 step4_analysis.py  5000 --time-trace --modes 12 24 -reg 1e5 7e4
python3 step4_analysis.py 10000 --time-trace --modes 23 48 -reg 8e4 3e4
python3 step4_analysis.py 20000 --time-trace --modes 44 94 -reg 2e5 1e4
python3 step4_analysis.py  5000 --species-integral --modes 12 17 24 -reg 1e5 6e5 7e4
python3 step4_analysis.py 10000 --species-integral --modes 23 33 48 -reg 8e4 4e4 3e4
python3 step4_analysis.py 20000 --species-integral --modes 44 62 94 -reg 2e5 2e4 1e4

# Generate / process snapshots for Tecplot animation.
python3 step4_analysis.py  5000 --save-snapshot -r 24 -reg 7e4
python3 step4_analysis.py 10000 --save-snapshot -r 48 -reg 3e4
python3 step4_analysis.py 20000 --save-snapshot -r 94 -reg 1e4
python3 step5_export.py --snapshot-type rom fom --trainsize  5000 -r 24 -reg 7e4 --variables 0 3 4
python3 step5_export.py --snapshot-type rom --trainsize 10000 -r 48 -reg 3e4 --variables 0 3 4
python3 step5_export.py --snapshot-type rom --trainsize 20000 -r 94 -reg 1e4 --variables 0 3 4


# =============================================================================
# =============================================================================
# =============================================================================

# History of commands for the JRSNZ paper.

'''
      |  r needed to exceed cumulative_energy(r)
  k   | .985 | .990 | .995 | .9975 | .999 | .9999
------|------|------|------|-------|------|--------
 5000 |   11 |   14 |   19 |    24 |   32 |    55
10000 |   22 |   28 |   37 |    47 |   62 |   108
20000 |   44 |   54 |   73 |    94 |  123 |   214
30000 |   68 |   83 |  112 |   144 |  188 |   326
40000 |   87 |  108 |  147 |   188 |  247 |   426
'''

# Get projected training data.
python3 step2_preprocess.py  5000 11 24
python3 step2_preprocess.py 10000 22 47 22
python3 step2_preprocess.py 20000 44 94 22
python3 step2_preprocess.py 30000 68    22
python3 step2_preprocess.py 40000 87    22

# Train selected models.
python step3_train.py --save-all  5000 -r  11 -reg  19744
python step3_train.py --save-all  5000 -r  24 -reg  24292
python step3_train.py --save-all 10000 -r  22 -reg  34610
python step3_train.py --save-all 10000 -r  47 -reg 232352
python step3_train.py --save-all 20000 -r  22 -reg  40089
python step3_train.py --save-all 20000 -r  44 -reg  29638
python step3_train.py --save-all 20000 -r  94 -reg  19440
python step3_train.py --save-all 30000 -r  22 -reg 205093
python step3_train.py --save-all 30000 -r  68 -reg  52878
python step3_train.py --save-all 40000 -r  22 -reg 107168
python step3_train.py --save-all 40000 -r  87 -reg  52898

# Optimal searching with bounds that work.
python step3_train.py  5000 --minimize -r 11 -reg 1.8e4 2.2e4 --margin 1.5
python step3_train.py  5000 --minimize -r 24 -reg 2.2e4 2.6e4 --margin 1.5
python step3_train.py 10000 --minimize -r 22 -reg 3.2e4 3.6e4 --margin 1.5
python step3_train.py 10000 --minimize -r 47 -reg 2.2e5 2.4e5 --margin 1.5
python step3_train.py 20000 --minimize -r 22 -reg 3.8e4 4.2e4 --margin 1.5
python step3_train.py 20000 --minimize -r 44 -reg 2.8e4 3.2e4 --margin 1.5
python step3_train.py 20000 --minimize -r 94 -reg 1.8e4 2.2e4 --margin 1.5
python step3_train.py 30000 --minimize -r 22 -reg 1.8e5 2.2e5 --margin 1.5
python step3_train.py 30000 --minimize -r 68 -reg 5.1e4 5.3e4 --margin 1.5
python step3_train.py 40000 --minimize -r 22 -reg 1.0e5 1.1e5 --margin 1.5
python step3_train.py 40000 --minimize -r 87 -reg 5.2e4 5.4e4 --margin 1.5

# Search figures
python step3_train.py  5000 --gridsearch -r 11 -reg 3e2 3e6 --regrange 100 --margin 1.5
python step3_train.py 10000 --gridsearch -r 22 31 47 -reg 3e2 3e6 --regrange 100 --margin 1.5

# Train models for mode bounding figure
python step3_train.py 10000 --save-all -r 47 -reg 2e5 4e5

# Generate snapshots for Tecplot figures.
python step5_export.py -type fom rom -snap 15000 20000 25000 -vars p T CH4 --trainsize 20000 -r 44 -reg 29638


# Plots for second AF review ==================================================

# Time traces
python3 step4_plot.py 10000 --time-traces --modes 22 --regularization 34610
python3 step4_plot.py 20000 --time-traces --modes 44 --regularization 29638
python3 step4_plot.py 30000 --time-traces --modes 68 --regularization 52878

# Animations: pressure, Temperature, CH4
python3 step5_export.py gems -vars p T CH4
python3 step5_export.py rom --trainsize 10000 -r 22 -reg 34610 -vars p T CH4
python3 step5_export.py rom --trainsize 20000 -r 44 -reg 29638 -vars p T CH4
python3 step5_export.py rom --trainsize 30000 -r 68 -reg 52878 -vars p T CH4
python3 poddeim.py --export -vars p T CH4

# Animations: velocities, specific volume
python3 step5_export.py gems -vars vx vy xi
python3 step5_export.py rom --trainsize 10000 -r 22 -reg 34610 -vars vx vy xi
python3 step5_export.py rom --trainsize 20000 -r 44 -reg 29638 -vars vx vy xi
python3 step5_export.py rom --trainsize 30000 -r 68 -reg 52878 -vars vx vy xi
python3 poddeim.py --export -vars vx vy xi

# Animations: other species
python3 step5_export.py gems -vars O2 H2O CO2
python3 step5_export.py rom --trainsize 10000 -r 22 -reg 34610 -vars O2 H2O CO2
python3 step5_export.py rom --trainsize 20000 -r 44 -reg 29638 -vars O2 H2O CO2
python3 step5_export.py rom --trainsize 30000 -r 68 -reg 52878 -vars O2 H2O CO2
python3 poddeim.py --export -vars O2 H2O CO2



'''
THINGS TO DO

* Clean up poddeim.py. Put it with plots.py on a separate branch after merging in other changes (branch from master).
'''

# =============================================================================
# =============================================================================
# =============================================================================

# History of commands for the *revised* JRSNZ paper.

'''
      |  r needed to exceed cumulative_energy(r)
  k   | .985 | .990 | .995 | .9975 | .999 | .9999
------|------|------|------|-------|------|--------
 5000 |   11 |   14 |   19 |    24 |   32 |    55
10000 |   22 |   28 |   37 |    47 |   62 |   108
20000 |   44 |   54 |   73 |    94 |  123 |   214
30000 |   68 |   83 |  112 |   144 |  188 |   326
40000 |   87 |  108 |  147 |   188 |  247 |   426
'''

# Get projected training data.
python3 step2_preprocess.py 10000 22 28  37  47
python3 step2_preprocess.py 20000 44 54  73  94
python3 step2_preprocess.py 30000 68 83 112 144

# Train selected models.
# python step3_train.py --minimize 10000 -r 37 -reg 1e4 1e6 --testsize 60000 --margin 1.05
python step3_train.py --save-all 10000 -r 37 -reg 131255

# python step3_train.py --minimize 20000 -r 73 -reg 1e4 1e5 --testsize 60000 --margin 1.05
python step3_train.py --save-all 20000 -r 73 -reg 42628

# python step3_train.py --minimize 30000 -r 112 -reg 1e4 1e5 --testsize 60000 --margin 1.05
python step3_train.py --save-all 30000 -r 112 -reg 20657


# PREVIOUS:
# k = 10,000, r = 22, reg = 34610 (WINNER)
# k = 20,000, r = 44, reg = 29638 (WINNER)
# k = 30,000, r = 68, reg = 52878 (WINNER)

####

# k = 10,000, r = 22, --trainsize 60,000,
### --margin 1.05 -> reg = 89942
### --margin 1.50 -> reg = 75350
### --margin 2.00 -> reg = 75324
# k = 10,000, r = 22, --trainsize 70,000
### --margin 1.05 -> reg = 89942
### --margin 1.50 -> reg = 76820
### --margin 2.00 -> reg = 76820
# k = 10,000, r = 22, --trainsize 80,000
### --margin 1.05 -> reg = 158075
### --margin 1.50 -> reg = 76819
### --margin 2.00 -> reg = 76819

# k = 10,000, r = 37, --trainsize 60,000,
### --margin 1.05 -> reg = 131255
### --margin 1.50 -> reg = 127969
### --margin 2.00 -> reg = 096328
# k = 10,000, r = 37, --trainsize 70,000
### --margin 1.05 -> reg = 323654
### --margin 1.50 -> reg = 132424
### --margin 2.00 -> reg = 131567
# k = 10,000, r = 37, --trainsize 80,000
### --margin 1.05 -> reg = FAILED
### --margin 1.50 -> reg = 322953
### --margin 2.00 -> reg = 236742


# k = 20,000, r = 44, --trainsize 60,000,
### --margin 1.05 -> reg = 28620
### --margin 1.50 -> reg = 28448
### --margin 2.00 -> reg = 28408
# k = 20,000, r = 44, --trainsize 70,000
### --margin 1.05 -> reg = 44072
### --margin 1.50 -> reg = 29638
### --margin 2.00 -> reg = 29591
# k = 20,000, r = 44, --trainsize 80,000
### --margin 1.05 -> reg = 75291
### --margin 1.50 -> reg = 52620
### --margin 2.00 -> reg = 52239

# k = 20,000, r = 73, --trainsize 60,000,
### --margin 1.05 -> reg = 17165
### --margin 1.50 -> reg = 16712
### --margin 2.00 -> reg = 16492
# k = 20,000, r = 73, --trainsize 70,000
### --margin 1.05 -> reg = 290616
### --margin 1.50 -> reg = 26356
### --margin 2.00 -> reg = 26167
# k = 20,000, r = 73, --trainsize 80,000
### --margin 1.05 -> reg = 680916
### --margin 1.50 -> reg = 39012 (this one is okay)
### --margin 2.00 -> reg = 38365


# k = 30,000, r = 68, --trainsize 60,000,
### --margin 1.05 -> reg = 25706
### --margin 1.50 -> reg = 25706
### --margin 2.00 -> reg = 25706
# k = 30,000, r = 68, --trainsize 70,000
### --margin 1.05 -> reg = 148969
### --margin 1.50 -> reg = 69651
### --margin 2.00 -> reg = 68102
# k = 30,000, r = 68, --trainsize 80,000
### --margin 1.05 -> reg = 251027
### --margin 1.50 -> reg = 239851
### --margin 2.00 -> reg = 186990

# k = 30,000, r = 112, --trainsize 60,000,
### --margin 1.05 -> reg = 20655
### --margin 1.50 -> reg = 20009
### --margin 2.00 -> reg = 14806
# k = 30,000, r = 112, --trainsize 70,000
### --margin 1.05 -> reg = 28360
### --margin 1.50 -> reg = 24316 (okayish)
### --margin 2.00 -> reg = 24152
# k = 30,000, r = 112, --trainsize 80,000
### --margin 1.05 -> reg = FAILED
### --margin 1.50 -> reg = 262810
### --margin 2.00 -> reg = 243877


# REVISON =====================================================================

'''
       |               ROM dimension r needed to exceed cumulative_energy(r) and corresponding data matrix column dimension d(r)
   k   |     .985      |      .990     |      .995      |      .9975      |       .999      |      .9999      |   .99999        |     .999999 
-------|---------------|---------------|----------------|-----------------|-----------------|-----------------|------------- ---|----------------
10,000 | r=22, d=  277 | r=27, d=  407 | r= 36, d=  704 | r= 46, d= 1,129 | r= 61, d= 1,954 | r=108, d= 5,996 | r=172 d= 15,052 | r=268 d= 36,316
20,000 | r=43, d=  991 | r=53, d=1,486 | r= 72, d=2,701 | r= 92, d= 4,372 | r=121, d= 7,504 | r=214, d=23,221 | r=342 d= 58,997 | r=524 d=138,076
30,000 | r=66, d=2,279 | r=82, d=3,487 | r=110, d=6,217 | r=141, d=10,154 | r=186, d=17,579 | r=326, d=53,629 | r=521 d=136,504 | r=804 d=324,416
'''

python step1_unpack.py /storage/combustion_gems_2d/rawdata

python step2_preprocess.py 10000  50
python step2_preprocess.py 20000 100
python step2_preprocess.py 30000 150

# python step3_train.py --minimize 10000  46  845 23172
# python step3_train.py --minimize 20000  92 8154  9617
# python step3_train.py --minimize 30000 141  258  7291

python step3_train.py --single 10000  46  845 23172
python step3_train.py --single 20000  92 8154  9617
python step3_train.py --single 30000 141  258  7291

python plots.py

# PSD for entire space/time domain is not very good
# PSD for spatially-averaged pressure *is* good, or pressure in one location

'''
import utils
maxenergy = .9999
d = lambda r: 1 + r + r*(r+1)//2 + 1
for k in [10000, 20000, 30000]:
    svdvals = utils.load_basis(k, -1)
    r0 = utils.roi.pre.cumulative_energy(svdvals, maxenergy)
    found = False
    r = 1
    while not found:
        if d(r) > r0:
            print(f"k = {k:d}, r = {r-1:d}")
            found = True
        r += 1

maxenergy = .9999
k = 10000, r = 13, λ = ( 287, 24528)
k = 20000, r = 19, λ = ( 350, 23636)
k = 30000, r = 24, λ = (4641, 54023)

python step3_train.py --single 10000 13  287 24528
python step3_train.py --single 20000 19  350 23636
python step3_train.py --single 30000 24 4641 54023

maxenergy = .99999
k = 10000, r = 17 Bad
k = 20000, r = 24, λ = ( 260, 31603) Meh
k = 30000, r = 30

maxenergy = .999999
k = 10000, r = 21
k = 20000, r = 30 Bad
k = 30000, r = 38

Old dimensions
k = 10000, r = 22, λ = ( 75, 29373)
k = 20000, r = 43, λ = (292, 18347)
k = 30000, r = 66, λ = (105, 27906)
'''

python step5_export.py fom rom --timeindex 5000 10000 15000 20000 25000 --variables T CH4 --trainsize 20000 --modes 43 --regularization 292 18347
python poddeim.py --export --timeindex 5000 10000 15000 20000 25000 --variables T CH4
