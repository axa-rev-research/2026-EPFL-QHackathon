    log_path = output_dir / f"log_data_L{L}"
    log = nk.logging.JsonLog(str(log_path))
    with timing.Timer() as timer:
        gs.run(
            n_iterations,
            out=log,
            obs={"ham": ha_p, "mz2": mz_p},
            step_size=10,
            #callback=SaveState(str(checkpoint_dir / f"L{L}"), save_every=50),
            timeit=True,
        )
        print(f"\n  Timing breakdown pour L={L}:")
        print(timer)
