"""Fixed EI/LCB comparison on identical recorded EI histories; no extra trials."""
import json
import pandas as pd
from .run import OUT,write_csv
from .environment import FrozenLegEnvironment
from lower_limb_sim.five_leg_mujoco_v1.model import load_frozen_benchmark_definition
from lower_limb_sim.five_leg_mujoco_v1.benchmark import build_leg_domain
from lower_limb_sim.dynamic_subject import DYNAMIC_SUBJECTS
from lower_limb_sim.parameter_estimator import baseline_template_from_dynamic_subject
from personalization.models.time_series_graybox import TimeSeriesFiveParameterGrayBoxAdapter,BranchBalancedE2GrayBoxEndpointAdapter
from personalization.models.physics_graybox import PhysicsSubjectModel
from personalization.models.residual_gp import PhysicsInformedResidualModel
from personalization.selectors.bo import ExpectedImprovementSelector,LowerConfidenceBoundSelector


def main():
    definition=load_frozen_benchmark_definition();history=pd.read_csv(OUT/'trial_history.csv');rows=[]
    for endpoint in ('E2','E0'):
        for leg in definition.legs:
            domain=build_leg_domain(leg);env=FrozenLegEnvironment(definition,leg,domain,endpoint)
            cls=BranchBalancedE2GrayBoxEndpointAdapter if endpoint=='E2' else TimeSeriesFiveParameterGrayBoxAdapter
            model=PhysicsInformedResidualModel(PhysicsSubjectModel(cls(domain,
                baseline_template=baseline_template_from_dynamic_subject(DYNAMIC_SUBJECTS['baseline']),
                L1=definition.thigh_length_m,L2=definition.cuff_distance_m)))
            subset=history[(history.endpoint==endpoint)&(history.leg_id==leg.leg_id)&(history.method=='MODEL_INFORMED_BO_EI')].sort_values('trial_index')
            observations=[]
            for item in subset.iloc[:3].itertuples():
                # Replay only the already recorded trial; selections are never executed here.
                observations.append(env.evaluate(domain.by_id(item.candidate_id),len(observations)+1))
                model.fit(observations)
                ei=ExpectedImprovementSelector(xi=0.).select_next(observations,domain,model)
                lcb=LowerConfidenceBoundSelector(name='MODEL_INFORMED_BO_LCB',kappa=1.5).select_next(observations,domain,model)
                expected=subset[subset.trial_index==len(observations)].iloc[0]
                assert ei.candidate.candidate_id==expected.candidate_id
                rows.append(dict(leg_id=leg.leg_id,endpoint=endpoint,history_trial_count=len(observations),
                    EI_beta=ei.candidate.beta,LCB_beta=lcb.candidate.beta,same_selection=ei.candidate==lcb.candidate,
                    EI_value=ei.acquisition_value,LCB_value=lcb.acquisition_value,
                    EI_metadata=ei.metadata,LCB_metadata=lcb.metadata))
    write_csv(OUT/'matched_history_ei_lcb.csv',rows)
    print('Matched-history selections:',sum(r['same_selection'] for r in rows),'/',len(rows))

if __name__=='__main__':main()
