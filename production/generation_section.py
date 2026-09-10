"""Generation narrative and deterministic illustrations from verified dashboard exports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.units import mm
from production.report_equations import equation
from src.dashboard_models import reference_name

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'studies/review_corrected_evaluation/results/dashboard/multimodel'
FAMILIES=['spline_pca','cvae','conditional_ae','unconditional_vae']
LABELS={'spline_pca':'Spline + PCA','cvae':'CVAE','conditional_ae':'CAE','unconditional_vae':'VAE'}
COLORS={'spline_pca':'#bb7226','cvae':'#27827b','conditional_ae':'#965382','unconditional_vae':'#365a8c'}


def generation_figures(assets):
    gen=pd.read_csv(DATA/'generated_features.csv');real=pd.read_csv(DATA/'empirical_features.csv')
    manifest=json.loads((DATA/'manifest.json').read_text());subject=manifest['example_subject']
    for dim in [3,8]:
        data={f:np.load(DATA/'examples'/f'{reference_name(f,dim)}.npz') for f in FAMILIES}
        recorded=data['cvae']['recorded'];shown=[recorded[:30],*[data[f]['generated'][:30] for f in FAMILIES]]
        allpoints=np.concatenate([x.reshape(-1,2) for x in shown]);lo=allpoints.min(0);hi=allpoints.max(0);pad=np.maximum((hi-lo)*.07,.15)
        fig,axs=plt.subplots(1,5,figsize=(9.5,2.7),sharex=True,sharey=True)
        for ax,curves,label,color in zip(axs,shown,['Recorded query',*[LABELS[f] for f in FAMILIES]],['black',*[COLORS[f] for f in FAMILIES]]):
            for curve in curves:ax.plot(*curve.T,color=color,alpha=.3,lw=.65)
            ax.set_title(label,fontsize=9);ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.grid(alpha=.15)
            ax.set_xlabel('x (tracker units)',fontsize=7);ax.tick_params(labelsize=7)
        axs[0].set_ylabel('y (tracker units)',fontsize=8)
        fig.suptitle(f'{subject} | n={dim} | first 30 query / generated trajectories',fontsize=10)
        fig.tight_layout();fig.savefig(assets/f'generation_paths_n{dim}.png',dpi=320,bbox_inches='tight');plt.close(fig)
    features=[('initiation_time_s','Initiation time (s)'),('movement_time_s','Movement time (s)'),
              ('peak_speed_tracker_units_s','Peak speed (tracker units/s)'),('curvature_index','Path / straight-line distance'),
              ('max_lateral_deviation','Lateral deviation (tracker units)'),('n_submovements','Speed-peak count')]
    fig,axs=plt.subplots(2,3,figsize=(8.4,5.2),sharey=True)
    for ax,(feature,label) in zip(axs.flat,features):
        values=np.sort(real.loc[real.subject==subject,feature].to_numpy());ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color='black',lw=1.8,label='Recorded query')
        for f in FAMILIES:
            values=np.sort(gen.loc[(gen.subject==subject)&(gen.model_family==f)&(gen.latent_dim==8),feature].to_numpy())
            ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=COLORS[f],lw=1.15,label=LABELS[f])
        ax.set_xlabel(label,fontsize=8);ax.set_ylim(0,1.03);ax.grid(alpha=.15);ax.tick_params(labelsize=8)
    for ax in axs[:,0]:ax.set_ylabel('Cumulative proportion',fontsize=8)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,fontsize=9)
    fig.suptitle(f'{subject} | n=8 | recorded and generated feature distributions',fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.97));fig.savefig(assets/'generation_ecdfs.png',dpi=300,bbox_inches='tight');plt.close(fig)
    means=pd.read_csv(DATA/'model_means.csv')
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.6))
    for ax,(metric,title) in zip(axs,[('mean_ks','Mean KS'),('energy_distance','Energy discrepancy'),('mmd_rbf','MMD squared')]):
        for f in FAMILIES:
            vals=means[(means.model_family==f)&means.latent_dim.isin([3,8])].sort_values('latent_dim')
            ax.plot([3,8],vals[metric],marker='o',lw=1.2,ms=4,color=COLORS[f],label=LABELS[f])
        ax.set_title(title,fontsize=9);ax.set_xticks([3,8]);ax.set_xlabel('Latent dimension');ax.grid(alpha=.15)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.12,1,1));fig.savefig(assets/'generation_scores.png',dpi=300,bbox_inches='tight');plt.close(fig)


def write_generation(a,assets):
    a.page();a.h('5 Generating participant-specific movement distributions')
    a.p('Accurate reconstruction shows that a code retains information about an observed movement. The fingerprint objective asks whether a code obtained from a participant\'s previous trials can also describe their other movements. We therefore evaluated generation using disjoint context and query trials from participants excluded from model development. Because generated movements have no unique recorded counterpart, this evaluation compared distributions rather than assigning a pointwise reconstruction error to each generated trajectory.')
    a.h('5.1 Context enrollment and stochastic generation',True)
    a.p('Within each held-out participant, trials were divided approximately equally into context and query sets, stratified by starting category and side. A fixed partition was shared across models. Each context trajectory was encoded, and its latent code contributed to the participant fingerprint. For variational models the code was the posterior mean; CAE used its deterministic encoding and spline + PCA used its PCA scores. The fingerprint was the mean of these context codes:')
    equation(a,assets,'fingerprint')
    a.p('Here C denotes the participant\'s context set and the fingerprint has n values. Trial variability was supplied by a shared covariance estimated from training participants, allowing multiple latent samples to be drawn around that fingerprint. For the neural models, this covariance combined variation in trial codes around each training participant\'s mean with the mean encoder posterior variance:')
    equation(a,assets,'covariance')
    a.p('The small diagonal term stabilizes sampling. For CAE, the posterior-variance contribution is approximately zero. Spline + PCA used the covariance of training scores after subtracting each training participant\'s mean, with the same diagonal stabilizer. Consequently, individual context trials determine the fingerprint center, while the covariance is shared. The n-value fingerprint does not independently estimate each participant\'s variability or covariance structure.')
    a.p('For each participant and fitted model, 120 latent samples were decoded into trajectories and durations. CVAE and CAE received task-condition vectors drawn from the participant\'s query-condition mixture; VAE ignored those vectors. Query task metadata supplied the mixture being evaluated, while query trajectories were withheld from fingerprint construction and generation. For spline + PCA, sampled scores were decoded through inverse PCA and the spline basis, and the timing Ridge received those scores and conditions. Its spatial decoder did not depend on conditions.')
    a.p('The generated outputs were therefore complete phase-normalized trajectories accompanied by predicted movement and initiation times. The evaluation reflects generation under the observed condition mixture. It does not establish extrapolation to new task conditions or adaptation across recording sessions.')
    a.h('5.2 Movement features',True)
    a.p('We extracted 11 timing, speed and geometry features (Table 3). Recorded features used the observed movement-onset-to-end segment. Generated segments used predicted initiation as a fraction of total duration, capped at 0.98, followed by resampling to 100 points. Movement duration was floored at 1 ms; initiation time at zero.')
    a.page();a.h('5.2 Movement features (continued)',True)
    a.table([['Feature','Definition and unit'],
        ['Initiation time','Target-motion onset to finger onset; seconds.'],
        ['Movement time','Finger onset to the final sample; seconds.'],
        ['Peak speed','Maximum norm of the position gradient; tracker units/s.'],
        ['Time to peak speed','Peak-speed sample index divided by the number of movement intervals; fraction.'],
        ['Path length','Sum of consecutive position-step lengths; tracker units.'],
        ['Straight-line distance','Distance from movement start to movement end; tracker units.'],
        ['Curvature index','Path length divided by straight-line distance; dimensionless.'],
        ['Maximum lateral deviation','Largest perpendicular distance to the start-to-end line; tracker units.'],
        ['Speed-peak count','Peaks with prominence at least 10% of the maximum and separation of about 50 ms; minimum reported count is one.'],
        ['Endpoint x; endpoint y','Two features: final movement coordinates relative to movement onset; tracker units.']],
        '<b>Table 3.</b> The 11 generated-feature endpoints. Speed-peak count is a heuristic descriptor and is distinct from the minimum-jerk component count evaluated later.',[48*mm,118*mm])
    a.p('Both recorded and generated movement segments passed through the same feature function. Positions were low-pass filtered with a cutoff of min(10 Hz, 0.45 times the effective sampling rate), and finite differences produced velocity. The effective rate was 99 divided by movement duration in seconds. The feature code was shared, although recorded onset and timing were measured and generated onset and timing were predicted. The recorded segments also inherited acquisition preprocessing; this comparison does not remove that upstream difference.')
    a.h('5.3 Distribution comparisons',True)
    a.p('For each participant and feature, the two-sample Kolmogorov-Smirnov statistic [3] measured the largest difference between the recorded-query and generated empirical cumulative distributions. The mean of the 11 statistics summarized marginal feature fidelity:')
    equation(a,assets,'ks')
    a.p('Lower values indicate closer distributions. We used KS as a descriptive distance, including for discrete peak counts; continuous-distribution KS p-values were not used to infer fidelity of that discrete feature. Matching marginal distributions alone does not ensure that relationships among features are reproduced.')
    a.p('Energy discrepancy and squared maximum mean discrepancy (MMD) [5] therefore compared complete 11-feature vectors. Features were centered and scaled using training data, with a scale floor of one for peak counts. The energy statistic was twice the average cross-sample Euclidean distance minus the two average within-sample distances:')
    equation(a,assets,'energy')
    a.p('These averages included all ordered pairs and within-sample diagonal zeros; the reported statistic was not square-rooted. MMD used a Gaussian kernel with bandwidth fixed from training data and excluded diagonal within-sample pairs:')
    equation(a,assets,'mmd')
    a.p('Here m and r are recorded and generated sample counts. The kernel was exp(-gamma times squared Euclidean distance), with gamma set to the inverse median positive squared distance among a fixed training subset of up to 512 trials. This unbiased MMD-squared estimator can be slightly negative in finite samples. All distance references were shared across models within a fold.')
    a.page();a.h('5.4 Generated-feature results',True)
    a.p('Distances were computed separately for each test participant, averaged across available neural training seeds and then across the 28 participants with equal weight. Table 4 presents the four compact-representation families at matched capacities. Spline + PCA contributes a single deterministic fit and fixed sampling realization per fold and dimension; neural results summarize seeds 42, 43 and 44.')
    means=pd.read_csv(DATA/'model_means.csv')
    rows=[['Model','n','Mean KS','Energy','MMD squared']]
    for n in [3,8]:
        for f in FAMILIES:
            row=means[(means.model_family==f)&(means.latent_dim==n)].iloc[0]
            rows.append([LABELS[f],str(n),*[f'{row[k]:.6f}' for k in ['mean_ks','energy_distance','mmd_rbf']]])
    a.table(rows,'<b>Table 4.</b> Cohort-average generated-feature discrepancies. Lower is better. All three metrics compare distributions; none is a trajectory reconstruction MSE.',[54*mm,12*mm,33*mm,33*mm,34*mm])
    a.figure(assets/'generation_scores.png','<b>Figure 7.</b> Generated-feature discrepancies at three and eight dimensions. Lines connect tested settings only and do not estimate performance at untested capacities.',62*mm)
    a.p('The reconstruction ranking did not carry over unchanged to generation. At n=3, CVAE had the lowest mean KS (0.277795) and energy discrepancy (0.745431) among these four families, while VAE had the lowest MMD-squared value (0.130298). At n=8, VAE had the lowest mean on all three distances: KS 0.215121, energy 0.411014 and MMD squared 0.063881. Spline + PCA retained its reconstruction advantage but had larger generation discrepancies than VAE and CVAE at n=8.')
    a.p('Numerical advantages were not uniformly significant. Within the same exploratory 140-comparison family used for reconstruction, the n=8 VAE-CVAE contrast survived Benjamini-Hochberg adjustment for energy (q=0.02575), but not for mean KS (q=0.13686) or MMD squared (q=0.05182). None of these three contrasts survived Holm adjustment across 140 tests. The results support an endpoint-specific numerical preference rather than universal dominance.')
    a.p('VAE n=8 was consequently selected as the dashboard\'s default for marginal-distribution exploration. All other supported references remain selectable. The choice uses this evaluated cohort and has no separate post-selection validation. Three-dimensional models remain available for the original compact-fingerprint question; n=16 was not evaluated in the matched study. A higher-capacity model might improve a particular endpoint, but the present data do not establish that outcome.')
    a.page();a.h('5.5 Recorded and generated movement examples',True)
    a.figure(assets/'generation_paths_n3.png','<b>Figure 8.</b> Recorded query trajectories and generated trajectories at n=3 for subject01. Each panel displays the first 30 available samples in its fixed ordering. Generated panels use context fingerprints, training-derived covariance and the seed-42 models. The recorded paths were not reconstruction inputs.',66*mm)
    a.figure(assets/'generation_paths_n8.png','<b>Figure 9.</b> The corresponding n=8 samples for the same participant. Axis limits are shared across models within each figure and include every displayed point; Figures 8 and 9 can have different limits. Axes are not drawn at equal physical scale. No generated sample was selected for resembling a recorded trial.',66*mm)
    a.p('The participant was selected as the first identifier in the analyzed cohort, independently of reconstruction or distribution scores. This participant was held out in fold 2. Each model generated 120 trajectories for evaluation; 30 are displayed to limit overplotting. The panels illustrate the range of decoded paths, while Table 4 summarizes the entire cohort and all evaluated seeds. Visual resemblance of individual examples is not evidence that two stochastic samples should coincide point by point.')
    a.figure(assets/'generation_ecdfs.png','<b>Figure 10.</b> Six feature distributions for subject01 at n=8: recorded query trials and all 120 generated samples per model, using seed 42. These fixed display features cover timing, speed, geometry and peak count. Vertical separation between two cumulative curves illustrates a distribution discrepancy; Table 4 averages all 11 features and all participants.',90*mm)
    a.p('These comparisons assess generation from personal context, but they do not yet isolate the contribution of the personal fingerprint. The next analysis holds the decoder and sampling procedure fixed while replacing the participant\'s context fingerprint with population and wrong-participant controls.')
