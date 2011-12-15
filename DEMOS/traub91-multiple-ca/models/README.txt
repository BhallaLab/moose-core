README.txt            - This file.
CA1.p                 - Traub's original CA1 model.
CA3_original.p        - Traub's original CA3 model.
CA3_original_neat.p   - Traub's original CA3 model, cleaned up, and
                        objects renamed so that they work with the
                        modified script for testing multiple Ca-pools.
CA3_passive.p         - All channels and Ca pools removed.
CA3_reference.p       - Reference model with 1 Ca-pool, 1 Ca-channel
                        and 1 Ca-dependent channel per compartment.
                        
                        Vm -> Ca -> Vm feedback loop cut by
                        silencing Ca-dependent channels:
                            - K_AHP silenced by setting density to 0.0.
                            - K_C deleted.
                        
                        This will be used for:
                            - Checking MOOSE against GENESIS
                            - Checking more complicated models (below)
                              against this one.
                        
                        This file is derived from 'CA3_original_neat.p'
CA3_A0.p              - This file and the remaining files below are the
                        main models used in the test simulations.
                        All are derived from 'CA3_reference.p'. More
                        info in the parent directory's README.txt.
CA3_A1.p
CA3_B0.p
CA3_B1.p
CA3_C0.p
CA3_C1.p
CA3_D0.p
CA3_D1.p
