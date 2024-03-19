        # 평가
        test_loader = dataloader.get_loader(rf_path=rf_path, tmax_path=tmax_path, tmin_path=tmin_path, runoff_path=runoff_path,
        start_date = test_start_date, end_date = test_end_date, past_length = past_length, pred_length = pred_length, batch_size = 1, run_dir = run_dir, loader_type = "test")
        torch.save(test_loader, Path(run_dir) / "test_loader.pt")

        # make directory folder
        
        test_path =  Path(run_dir) / "test"
        test_path.mkdir(parents=True, exist_ok=True)
        evaluation.evaluation(model, test_loader, writer, Path(run_dir) / "test", device)
        print("evaluation is done")