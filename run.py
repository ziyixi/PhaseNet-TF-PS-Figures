import sys

from phasenettfps.scripts import *

scripts_mapper = {
    "manuscript_event_station_distribution": manuscript_event_station_distribution_main,
    "manuscript_manual_picks_ptf_example": manuscript_manual_picks_ptf_example_main,
    "manuscript_timediff_vs_distance": manuscript_timediff_vs_distance_main,
    "manuscript_ptf_predictions_ptf_example": manuscript_ptf_predictions_ptf_example_main,
    "manuscript_beamforming_example": manuscript_beamforming_example_main,
    "manuscript_mcmc_histogram": manuscript_mcmc_histogram_main,
    "manuscript_prior_posterior_distribution": manuscript_prior_posterior_distribution_main,
    "manuscript_ps_residual": manuscript_ps_residuals_main,
    "manuscript_compare_manual_final_catalogs": manuscript_compare_manual_final_catalogs_main,
    "manuscript_ps_residuals": manuscript_ps_residuals_main,
    "manuscript_final_map_compare": manuscript_final_map_compare_main,
}


def main():
    if len(sys.argv) == 2:
        if sys.argv[1] == "all" or (sys.argv[1] in scripts_mapper):
            if sys.argv[1] != "all":
                scripts_mapper[sys.argv[1]]()
            else:
                for key in scripts_mapper:
                    print(f"Plot {key} now...")
                    scripts_mapper[key]()
        else:
            raise Exception(f"scripts {sys.argv[1]} is not supported!")
    else:
        raise Exception("correct format: python run.py [script name]")


if __name__ == "__main__":
    main()
