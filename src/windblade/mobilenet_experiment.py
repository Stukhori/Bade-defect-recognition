"""Canonical Phase 6 MobileNetV3-Small experiment using Phase 5 shared infrastructure."""

from __future__ import annotations
from copy import deepcopy
from importlib import metadata
import json, hashlib, shutil
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch, yaml
import matplotlib.pyplot as plt

from windblade.config import ResolvedConfig, load_config
from windblade.data.processed import LABELS, json_text, read_csv, validate_processed_dataset
from windblade.data.subsets import validate_training_subsets
from windblade.deep.checkpoints import load_checkpoint, save_checkpoint, state_dict_fingerprint
from windblade.deep.dataset import WTBDCropDataset, balanced_class_weights, make_loader, split_rows
from windblade.deep.determinism import resolve_device, seed_torch
from windblade.deep.mobilenet import EXPECTED_MOBILENET_PARAMETERS, WEIGHT_ENUM, load_official_model, model_from_official_state
from windblade.deep.training import aggregate_seed_metrics, hyperparameter_grid, inference_latency, run_epoch, select_candidate, train_with_validation
from windblade.environment import capture_environment, capture_git_provenance
from windblade.evaluation.reporting import plot_confusion, write_csv, write_matrix_csv
from windblade.resnet_experiment import EXPECTED_FINGERPRINT, HISTORY_FIELDS, PREDICTION_FIELDS, _grid_fingerprint, _plot_curves, _plot_tuning, _publish, _records, _safe_clean, _write_training, validate_resnet18_results
from windblade.traditional import validate_traditional_results
from windblade.utils import atomic_write_text, format_utc, utc_now


class MobileNetExperimentError(RuntimeError): pass


def verify_start_gate(config: ResolvedConfig, root: Path) -> dict[str, Any]:
    data = config.as_dict(); failures = []
    parity = load_config(root / "configs/resnet18_baseline.yaml").as_dict()
    shared = (("dataset", "processed_version"), ("dataset", "processed_fingerprint"), ("classes", "order"), ("input", "size"), ("input", "augmentation"), ("input", "normalization"), ("training", "batch_size"), ("training", "validation_batch_size"), ("training", "max_epochs"), ("training", "patience"), ("training", "min_delta"), ("training", "optimizer"), ("training", "betas"), ("training", "eps"), ("training", "class_weight"), ("search", "seed"), ("search", "learning_rates"), ("search", "weight_decays"), ("final", "seeds"), ("selection", "primary_metric"))
    for section, field in shared:
        if data[section][field] != parity[section][field]: failures.append(f"protocol parity {section}.{field}")
    if data["project"]["phase"] != 6 or data["model"]["architecture"] != "mobilenet_v3_small" or data["model"]["weights"] != "IMAGENET1K_V1" or int(data["model"]["expected_total_parameters"]) != EXPECTED_MOBILENET_PARAMETERS: failures.append("MobileNet identity")
    if failures: raise MobileNetExperimentError("Phase 6 configuration gate failed: " + ", ".join(failures))
    p3 = validate_processed_dataset(config, root); validate_training_subsets(config, root)
    p4 = validate_traditional_results(load_config(root / "configs/traditional_baselines.yaml"), root)
    p5 = validate_resnet18_results(load_config(root / "configs/resnet18_baseline.yaml"), root)
    if p3["processed_dataset_fingerprint"] != EXPECTED_FINGERPRINT: raise MobileNetExperimentError("Phase 3 fingerprint changed")
    return {"phase3": p3, "phase4": p4, "phase5": p5}


def _metadata(seed: int, result: Mapping[str, Any], lr: float, wd: float, commit: str | None) -> dict[str, Any]:
    return {"architecture":"torchvision_mobilenet_v3_small","class_count":6,"class_order":list(LABELS),"pretrained_weight_enum":WEIGHT_ENUM,"seed":seed,"epoch":result["best_epoch"],"learning_rate":lr,"weight_decay":wd,"processed_dataset_fingerprint":EXPECTED_FINGERPRINT,"git_commit":commit,"validation_metrics":result["best_validation_metrics"]}


def _freeze(path: Path, selected: Mapping[str, Any], grid_fp: str, pretrained_fp: str, weights: torch.Tensor, commit: str | None) -> dict[str, Any]:
    record={"schema_version":"1.0","phase":6,"processed_dataset_fingerprint":EXPECTED_FINGERPRINT,"class_order":list(LABELS),"model":{"architecture":"torchvision_mobilenet_v3_small","pretrained_weight_enum":WEIGHT_ENUM,"pretrained_mobilenet_fingerprint":pretrained_fp,"num_classes":6,"fine_tune":"all","total_parameters":EXPECTED_MOBILENET_PARAMETERS},"input":{"size":[224,224],"normalization":{"mean":[.485,.456,.406],"std":[.229,.224,.225]},"augmentation":"none"},"training":{"optimizer":"AdamW","learning_rate":selected["learning_rate"],"weight_decay":selected["weight_decay"],"betas":[.9,.999],"eps":1e-8,"batch_size":32,"validation_batch_size":64,"max_epochs":30,"patience":6,"min_delta":1e-4,"class_weight_formula":"N/(6*N_c), train only","class_weights":weights.tolist(),"scheduler":None,"mixed_precision":False,"num_workers":0},"selection":{"tuning_seed":17,"metric":"validation_macro_f1","validation_macro_f1":selected["validation_macro_f1"],"validation_balanced_accuracy":selected["validation_balanced_accuracy"],"validation_macro_recall":selected["validation_macro_recall"],"numeric_tolerance":1e-12,"grid_fingerprint":grid_fp},"final_seeds":[17,29,43],"git_commit":commit,"frozen_before_test":True,"frozen_utc":format_utc(utc_now())}
    atomic_write_text(path, yaml.safe_dump(record,sort_keys=True)); return record


def run_mobilenet(config: ResolvedConfig, repository_root: str|Path) -> dict[str,Any]:
    root=Path(repository_root).resolve(); data=config.as_dict(); gate=verify_start_gate(config,root); section=data["mobilenet"]
    output=root/section["output_root"]; summary=root/section["summary_root"]; figures=root/section["figures_root"]
    _safe_clean(output,root/"experiments/results"); _safe_clean(figures,root/"figures")
    commit=capture_git_provenance(root)["git_commit"]; processed=root/"data/processed/wtbd_crops_v1"; partitions=split_rows(read_csv(processed/"manifest.csv"))
    datasets={name:WTBDCropDataset(rows,processed,verify_hashes=True) for name,rows in partitions.items()}
    for dataset in datasets.values():
        for index in range(len(dataset)): dataset[index]
    weights=balanced_class_weights(partitions["train"]); seed_torch(17); official,pretrained=load_official_model(); state=deepcopy(official.state_dict()); del official
    pretrained.update({"torch_version":torch.__version__,"torchvision_version":metadata.version("torchvision")}); atomic_write_text(output/"pretrained_model.json",json_text(pretrained)); device=resolve_device(data["runtime"]["device"])
    tuning=[]
    for index,candidate in enumerate(hyperparameter_grid(),1):
        seed_torch(17); model=model_from_official_state(state,seed=17)
        result=train_with_validation(model,make_loader(datasets["train"],batch_size=32,shuffle=True,seed=17),make_loader(datasets["validation"],batch_size=64,shuffle=False,seed=17),device=device,class_weights=weights,learning_rate=candidate["learning_rate"],weight_decay=candidate["weight_decay"])
        run_root=output/"tuning"/f"config_{index:02d}"; _write_training(run_root,result); checkpoint=save_checkpoint(run_root/"best_state_dict.pt",result["best_state_dict"],_metadata(17,result,candidate["learning_rate"],candidate["weight_decay"],commit))
        tuning.append({"candidate_id":f"config_{index:02d}",**candidate,"best_epoch":result["best_epoch"],"validation_macro_f1":result["best_validation_metrics"]["macro_f1"],"validation_balanced_accuracy":result["best_validation_metrics"]["balanced_accuracy"],"validation_macro_recall":result["best_validation_metrics"]["macro_recall"],"training_seconds":result["training_seconds"],"checkpoint_fingerprint":checkpoint["checkpoint_fingerprint"],"selected":False})
    selected=select_candidate(tuning); selected["selected"]=True
    for row in tuning: row["selected"]=row["candidate_id"]==selected["candidate_id"]
    grid_fp=_grid_fingerprint(tuning); write_csv(output/"tuning/grid_results.csv",tuning,tuple(tuning[0])); frozen=_freeze(root/section["frozen_config"],selected,grid_fp,pretrained["pretrained_mobilenet_fingerprint"],weights,commit); atomic_write_text(output/"frozen/selected_hyperparameters.json",json_text(frozen)); _plot_tuning(tuning,figures/"tuning_validation_macro_f1.png", "MobileNetV3-Small")
    finals={}
    for seed in (17,29,43):
        seed_torch(seed); model=model_from_official_state(state,seed=seed); result=train_with_validation(model,make_loader(datasets["train"],batch_size=32,shuffle=True,seed=seed),make_loader(datasets["validation"],batch_size=64,shuffle=False,seed=seed),device=device,class_weights=weights,learning_rate=float(selected["learning_rate"]),weight_decay=float(selected["weight_decay"])); run_root=output/"final"/f"seed_{seed}"; _write_training(run_root,result); result["checkpoint"]=save_checkpoint(run_root/"best_state_dict.pt",result["best_state_dict"],_metadata(seed,result,float(selected["learning_rate"]),float(selected["weight_decay"]),commit)); finals[seed]=result; _plot_curves(result["history"],seed,figures/f"training_curves_seed{seed}.png")
    for seed,result in finals.items():
        path=output/"final"/f"seed_{seed}"/"best_state_dict.pt"; loaded,_=load_checkpoint(path,expected_dataset_fingerprint=EXPECTED_FINGERPRINT,expected_architecture="torchvision_mobilenet_v3_small"); result["model"].load_state_dict(loaded); criterion=torch.nn.CrossEntropyLoss(weight=weights.to(device)); _,metrics,records=run_epoch(result["model"],make_loader(datasets["test"],batch_size=64,shuffle=False,seed=seed),criterion,device); run_root=path.parent; atomic_write_text(run_root/"test_metrics.json",json_text(metrics)); write_csv(run_root/"test_predictions.csv",_records(records),PREDICTION_FIELDS); write_matrix_csv(run_root/"confusion_matrix_counts.csv",metrics["confusion_matrix_counts"],LABELS); write_matrix_csv(run_root/"confusion_matrix_normalized.csv",metrics["confusion_matrix_row_normalized"],LABELS); plot_confusion(metrics["confusion_matrix_row_normalized"],LABELS,figures/f"confusion_seed{seed}.png",normalized=True); result["test_metrics"]=metrics; result["test_records"]=records
    aggregate=aggregate_seed_metrics([finals[s]["test_metrics"] for s in (17,29,43)]); atomic_write_text(output/"aggregate/test_summary.json",json_text(aggregate)); write_matrix_csv(output/"aggregate/mean_normalized_confusion.csv",aggregate["mean_normalized_confusion"],LABELS); plot_confusion(aggregate["mean_normalized_confusion"],LABELS,figures/"confusion_mean_normalized.png",normalized=True)
    seed_rows=[]
    for seed,result in finals.items():
        m=result["test_metrics"]; seed_rows.append({"seed":seed,"best_epoch":result["best_epoch"],"validation_macro_f1":result["best_validation_metrics"]["macro_f1"],**{k:m[k] for k in ("macro_f1","balanced_accuracy","accuracy","macro_precision","macro_recall")},"training_seconds":result["training_seconds"],"epochs_executed":result["epochs_executed"],"average_seconds_per_epoch":result["training_seconds"]/result["epochs_executed"],"checkpoint_fingerprint":result["checkpoint"]["checkpoint_fingerprint"],"checkpoint_bytes":result["checkpoint"]["checkpoint_bytes"]})
    write_csv(output/"aggregate/per_seed_summary.csv",seed_rows,tuple(seed_rows[0])); class_rows=[{"class_id":i,"class_label":label,**{f"{metric}_{stat}":aggregate["per_class"][label][metric][stat] for metric in ("precision","recall","f1") for stat in ("mean","sample_sd")}} for i,label in enumerate(LABELS)]; write_csv(output/"aggregate/per_class_summary.csv",class_rows,tuple(class_rows[0]))
    timing=inference_latency(finals[17]["model"],datasets["test"][0]["image"],device); efficiency={"device":str(device),"total_parameters":EXPECTED_MOBILENET_PARAMETERS,"trainable_parameters":EXPECTED_MOBILENET_PARAMETERS,"checkpoint_bytes":finals[17]["checkpoint"]["checkpoint_bytes"],"inference":timing,"training":[{k:r[k] for k in ("seed","training_seconds","epochs_executed","best_epoch","average_seconds_per_epoch")} for r in seed_rows]}; atomic_write_text(output/"aggregate/efficiency.json",json_text(efficiency)); _write_comparisons(root,output,figures,aggregate,efficiency)
    seed_torch(17); repro_model=model_from_official_state(state,seed=17); repro=train_with_validation(repro_model,make_loader(datasets["train"],batch_size=32,shuffle=True,seed=17),make_loader(datasets["validation"],batch_size=64,shuffle=False,seed=17),device=device,class_weights=weights,learning_rate=float(selected["learning_rate"]),weight_decay=float(selected["weight_decay"])); criterion=torch.nn.CrossEntropyLoss(weight=weights.to(device)); _,repro_metrics,repro_records=run_epoch(repro_model,make_loader(datasets["test"],batch_size=64,shuffle=False,seed=17),criterion,device); scientific=lambda h:[{k:v for k,v in row.items() if k!="elapsed_seconds"} for row in h]; reproducibility={"status":"PASS","best_epoch_identical":repro["best_epoch"]==finals[17]["best_epoch"],"validation_predictions_identical":_records(repro["best_validation_records"])==_records(finals[17]["best_validation_records"]),"test_predictions_identical":_records(repro_records)==_records(finals[17]["test_records"]),"test_metrics_identical":repro_metrics==finals[17]["test_metrics"],"scientific_history_identical":scientific(repro["history"])==scientific(finals[17]["history"]),"checkpoint_fingerprint_identical":state_dict_fingerprint(repro["best_state_dict"])==finals[17]["checkpoint"]["checkpoint_fingerprint"]}; atomic_write_text(output/"reproducibility.json",json_text(reproducibility));
    if not all(v for k,v in reproducibility.items() if k!="status"): raise MobileNetExperimentError("seed-17 reproducibility failed")
    environment=capture_environment(str(device),root); manifest={"status":"completed","phase":6,"result_id":section["result_id"],"phase5_start_gate":gate["phase5"],"processed_dataset_fingerprint":EXPECTED_FINGERPRINT,"sample_counts":{k:len(v) for k,v in datasets.items()},"class_order":list(LABELS),"class_weights":weights.tolist(),"tuning_candidates":4,"tuning_test_evaluations":0,"selected_candidate":selected["candidate_id"],"grid_fingerprint":grid_fp,"frozen_before_test":True,"final_seeds":[17,29,43],"final_test_evaluations":3,"reproducibility_test_evaluations":1,"parameter_count":EXPECTED_MOBILENET_PARAMETERS,"pretrained_mobilenet_fingerprint":pretrained["pretrained_mobilenet_fingerprint"],"no_augmentation":True,"resnet_retrained":False,"data_efficiency_started":False,"robustness_started":False,"phase7_started":False,"environment":environment}; atomic_write_text(output/"manifest.json",json_text(manifest)); atomic_write_text(output/"resolved_config.yaml",config.to_yaml()); _publish(output,summary); return {"status":"PASS","selected":selected,"aggregate":aggregate["overall"],"manifest":manifest}


def validate_mobilenet_results(config: ResolvedConfig, repository_root: str|Path) -> dict[str,Any]:
    root=Path(repository_root).resolve(); verify_start_gate(config,root); summary=root/config.as_dict()["mobilenet"]["summary_root"]; manifest=json.loads((summary/"manifest.json").read_text())
    if manifest["status"]!="completed" or manifest["parameter_count"]!=EXPECTED_MOBILENET_PARAMETERS or manifest["tuning_candidates"]!=4 or manifest["tuning_test_evaluations"]!=0 or manifest["final_test_evaluations"]!=3 or manifest["phase7_started"] or manifest["resnet_retrained"]: raise MobileNetExperimentError("Phase 6 manifest invalid")
    if json.loads((summary/"reproducibility.json").read_text())["status"]!="PASS": raise MobileNetExperimentError("Phase 6 reproducibility failed")
    required={"tuning_validation_macro_f1.png","training_curves_seed17.png","training_curves_seed29.png","training_curves_seed43.png","confusion_seed17.png","confusion_seed29.png","confusion_seed43.png","confusion_mean_normalized.png","mobilenet_seed_variability.png","clean_model_macro_f1_comparison.png","cnn_accuracy_vs_parameters.png","cnn_accuracy_vs_latency.png"}
    if {p.name for p in (root/config.as_dict()["mobilenet"]["figures_root"]).glob("*.png")} != required: raise MobileNetExperimentError("Phase 6 figure set invalid")
    return {"status":"PASS","result_id":manifest["result_id"],"pretrained_mobilenet_fingerprint":manifest["pretrained_mobilenet_fingerprint"]}


def _write_comparisons(root:Path, output:Path, figures:Path, aggregate:Mapping[str,Any], efficiency:Mapping[str,Any])->None:
    p5=root/"experiments/summaries/phase5_resnet18_v1"; ragg=json.loads((p5/"aggregate/test_summary.json").read_text()); reff=json.loads((p5/"aggregate/efficiency.json").read_text())
    comparison={"parameter_reduction":1-EXPECTED_MOBILENET_PARAMETERS/reff["total_parameters"],"model_size_reduction":1-efficiency["checkpoint_bytes"]/reff["checkpoint_bytes"],"latency_reduction":1-efficiency["inference"]["median_seconds"]/reff["inference"]["median_seconds"],"macro_f1_performance_retention":aggregate["overall"]["macro_f1"]["mean"]/ragg["overall"]["macro_f1"]["mean"],"macro_f1_absolute_difference":aggregate["overall"]["macro_f1"]["mean"]-ragg["overall"]["macro_f1"]["mean"],"resnet_source":"experiments/summaries/phase5_resnet18_v1"}; atomic_write_text(output/"aggregate/comparison_to_resnet.json",json_text(comparison))
    rows=[]
    for family,name in (("hog","HOG + SVM"),("lbp","LBP + SVM")):
        m=json.loads((root/f"experiments/summaries/phase4_traditional_v1/{family}/test_metrics.json").read_text()); e=json.loads((root/f"experiments/summaries/phase4_traditional_v1/{family}/efficiency.json").read_text()); rows.append({"method":name,"macro_f1":m["macro_f1"],"macro_f1_sample_sd":"","balanced_accuracy":m["balanced_accuracy"],"accuracy":m["accuracy"],"parameters":"","model_size_bytes":e["model_size_bytes"],"device":"cpu","median_latency_seconds":e["combined_median_seconds_per_image"]})
    rows += [{"method":"ResNet-18","macro_f1":ragg["overall"]["macro_f1"]["mean"],"macro_f1_sample_sd":ragg["overall"]["macro_f1"]["sample_sd"],"balanced_accuracy":ragg["overall"]["balanced_accuracy"]["mean"],"accuracy":ragg["overall"]["accuracy"]["mean"],"parameters":reff["total_parameters"],"model_size_bytes":reff["checkpoint_bytes"],"device":reff["device"],"median_latency_seconds":reff["inference"]["median_seconds"]},{"method":"MobileNetV3-Small","macro_f1":aggregate["overall"]["macro_f1"]["mean"],"macro_f1_sample_sd":aggregate["overall"]["macro_f1"]["sample_sd"],"balanced_accuracy":aggregate["overall"]["balanced_accuracy"]["mean"],"accuracy":aggregate["overall"]["accuracy"]["mean"],"parameters":efficiency["total_parameters"],"model_size_bytes":efficiency["checkpoint_bytes"],"device":efficiency["device"],"median_latency_seconds":efficiency["inference"]["median_seconds"]}]
    write_csv(output/"aggregate/clean_model_comparison.csv",rows,tuple(rows[0])); names=[r["method"] for r in rows]; vals=[float(r["macro_f1"]) for r in rows]; errs=[float(r["macro_f1_sample_sd"] or 0) for r in rows]; fig,ax=plt.subplots(figsize=(9,5));ax.bar(names,vals,yerr=errs,capsize=4);ax.set_ylim(0,1);ax.set_ylabel("Test macro-F1");ax.set_title("Frozen clean-model comparison (CNN error bars: seed SD)");fig.tight_layout();fig.savefig(figures/"clean_model_macro_f1_comparison.png",dpi=160);plt.close(fig)
    seeds=read_csv(output/"aggregate/per_seed_summary.csv");fig,ax=plt.subplots(figsize=(8,5));v=[float(r["macro_f1"]) for r in seeds];ax.bar([r["seed"] for r in seeds],v);ax.axhline(np.mean(v),color="#e07a1f",linestyle="--");ax.set_ylim(0,1);ax.set_ylabel("Test macro-F1");ax.set_title("MobileNetV3-Small seed variability");fig.tight_layout();fig.savefig(figures/"mobilenet_seed_variability.png",dpi=160);plt.close(fig)
    for xkey,xlabel,filename in (("parameters","Parameters","cnn_accuracy_vs_parameters.png"),("median_latency_seconds","Median forward latency (seconds)","cnn_accuracy_vs_latency.png")):
        cnn=rows[-2:];fig,ax=plt.subplots(figsize=(7,5));ax.scatter([float(r[xkey]) for r in cnn],[float(r["accuracy"]) for r in cnn]);
        for r in cnn: ax.annotate(r["method"],(float(r[xkey]),float(r["accuracy"])))
        ax.set_xlabel(xlabel);ax.set_ylabel("Mean test accuracy");ax.set_ylim(0,1);ax.set_title("Frozen CNN efficiency comparison");fig.tight_layout();fig.savefig(figures/filename,dpi=160);plt.close(fig)
