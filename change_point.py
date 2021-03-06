import os,sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import pickle
import scipy.stats as st
import theano.tensor as tt
import pdb

plt.style.use('bmh')

def get_referral_data(filename):
    referrals = pd.read_csv(filename, sep='|')

    referrals['dater'] =  pd.to_datetime(referrals['dater'], infer_datetime_format=True)

    referrals['Quantity'] = 1

    referrals.drop(['REFERRAL_KEY', 'patient', 'regdate', 'sex', 'age', 'planname', 'plantype', 'priority_', 'ref_type', 'created_by', 'ppl', 'site_name', 'cpt1',
           'cpt2', 'cpt3', 'cpt4'], axis = 1, inplace=True)

    referrals['self_ref'] = referrals['ref_spec'] == referrals['ref_to_spec']
    return referrals

def print_graph_doc(ref_prov):
    fig = plt.figure(figsize=(12.5,5))
    ax = fig.add_subplot(111)

    ax.plot(range(len(expecrefs[ref_prov])), expecrefs[ref_prov], lw=4, color="#E24A33",
             label="expected number of referrals by week")
    ax.set_xlim(0, len(spec_refcounts[ref_prov]))
    ax.set_xlabel("Week")
    ax.set_ylabel("Expected # self-referrals")
    ax.set_title("Changes in Self-Referral for {}".format(ref_prov))
    ax.set_ylim(0, (np.max(spec_refcounts[ref_prov]) * 1.03))
    ax.bar(range(len(spec_refcounts[ref_prov])), spec_refcounts[ref_prov], color="#348ABD", alpha=0.65,label="observed self-referrals")

    ax.legend(loc="upper left")
    plt.show();
    #fig.savefig("/imgs/{}.png".format(ref_prov));

def find_self_ref_increases_for_spec(refcounts_main, stats_table, expecrefs, provs_and_specs, specialty=None):
    ## takes referrals data with "dater", "self_ref", "ref_spec" and returns potential change increases in
    ## change points.
    if specialty == None:
        provs_and_specs = provs_and_specs
    else:
        provs_and_specs = provs_and_specs[provs_and_specs['ref_spec'].isin(specialty)]
    provs = list(set(provs_and_specs['ref_prov']))
    length = len(provs)
    counter = 0
    for spec in specialty:
        provs = list(set(provs_and_specs[provs_and_specs['ref_spec']==spec]['ref_prov']))
        for idx, prov in enumerate(provs):
            counter += 1
            print('{0:0.4f} complete'.format(counter/length))
            ## assign lambdas and tau to stochastic variables
            refcounts = np.array(refcounts_main.loc[np.in1d(refcounts_main['ref_prov'], prov),'self_ref'])
            n_refcounts = len(refcounts)

            with pm.Model() as model:
                alpha = 1.0/refcounts.mean()  # Recall count_data is the
                                               # variable that holds our txt counts
                lambda_1 = pm.Exponential("lambda_1", alpha)
                lambda_2 = pm.Exponential("lambda_2", alpha)
                tau = pm.DiscreteUniform("tau", lower=0, upper=n_refcounts)

            ## create a combined function for lambda (it is still a RV)
            with model:
                idx = np.arange(n_refcounts) # Index
                lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)

            ## combine the data with our proposed data generation scheme
            with model:
                observation = pm.Poisson("obs", lambda_, observed=refcounts)

            ## inference
            with model:
                step = pm.Metropolis()
                trace = pm.sample(25, tune=2500,step=step)

            lambda_1_samples = trace['lambda_1']
            lambda_2_samples = trace['lambda_2']
            tau_samples = trace['tau']

            N = tau_samples.shape[0]
            expected_refs_per_week = np.zeros(n_refcounts)
            for week in range(0, n_refcounts):
                ix = week < tau_samples
                expected_refs_per_week[week] = (lambda_1_samples[ix].sum()
                                               + lambda_2_samples[~ix].sum()) / N

            expecrefs[prov] = expected_refs_per_week
            stats_table.loc[prov, 'specialty'] = spec
            stats_table.loc[prov, 'tau_mean'] = np.mean(tau_samples)
            stats_table.loc[prov, 'tau_std'] = np.std(tau_samples)
            stats_table.loc[prov, 'mean1'] = np.mean(lambda_1_samples)
            stats_table.loc[prov, 'mean2'] = np.mean(lambda_2_samples)
            stats_table.loc[prov, 'mean_diff'] = st.ttest_ind(lambda_1_samples, lambda_2_samples)[1]
    return stats_table, expecrefs

if __name__ == '__main__':
    provs_and_specs = pd.read_csv('../data/provs_and_specs.csv', index_col=0)
    refcounts_main = pd.read_csv('../data/refcounts.csv', index_col=0)
    stats_table = pd.read_csv('../data/stats_table.csv', index_col=0)
    filename = '../data/expecrefs.p'
    expecrefs = pickle.load(open(filename, 'rb'))
    stats_table, expecrefs = find_self_ref_increases_for_spec(refcounts_main, stats_table, expecrefs, provs_and_specs, ['PSY', 'URO'])
    pickle.dump(expecrefs, open(filename, "wb"))
    stats_table.to_csv('../data/stats_table.csv')

    # df = referrals.groupby(['ref_prov', pd.Grouper(key='dater', freq='W-MON')])['self_ref'].sum().reset_index().sort_values('dater')
