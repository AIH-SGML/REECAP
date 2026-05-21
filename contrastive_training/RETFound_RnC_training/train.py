import argparse
import torch
import multiprocessing
from trainer import get_trainer
from data import load_data, create_loaders, create_RnC_loaders
from util.transforms import custom_transform_with_selective_augs
from util.utils import set_seed, validate_embeddings, save_embeddings_to_csv
import pandas as pd
from os.path import join
from util.utils import set_seed
import time
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--path_table_file", type=str, required=True, 
                        help="Path to the CSV file containing file paths and labels. The CSV should have columns 'file_path' and 'age'")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for the optimizer. Only used in training mode")
    parser.add_argument("--train_mode", type=str, default="supervised",
                        choices=["supervised", "rnc_original"],
                        help="supervised or rnc_original training mode. In rnc_original, the contrastive loss is used and the batch size should be at least 4")
    parser.add_argument("--num_epochs", type=int, default=50, 
                        help="Number of epochs to train. Only used in training mode")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Minimal batch_size for rnc_original is 4")
    parser.add_argument("--aug", type=str, default=None, choices=[None, "yes"],
                        help="Whether to apply data augmentation during training. Only used in training mode")
    parser.add_argument("--drop_path_rate", type=float, default=0, 
                        help="Drop path rate for the backbone. Only used in training mode")
    parser.add_argument("--pos_drop_rate", type=float, default=0,
                        help="Drop positional embeddings with this rate. Only used in training mode")
    parser.add_argument("--patch_drop_rate", type=float, default=0,
                        help="Drop patches with this rate. Only used in training mode")
    parser.add_argument("--mask_ratio", type=float, default=0,
                        help="Mask ratio for the input. Only used in training mode")
    parser.add_argument("--zdim", type=int, default=1024,
                        help="Dimensionality of the embedding space")
    parser.add_argument("--temperature", type=float, default=0.5, 
                        help="Temperature for contrastive loss. Only used in rnc_original training mode")
    parser.add_argument("--freeze_layers_until_epoch", type=int, default=0,                    
                        help="Number of epochs until which to freeze the backbone and only train the head and fn norm layers. For supervised training mode")
    parser.add_argument("--leave_fold_out", type=int, default=0, 
                        help="Which fold to leave out for testing")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory for saving models, embeddings, and predictions")
    parser.add_argument("--resume", action="store_true", 
                        help="Option to resume training from the last checkpoint")
    parser.add_argument("--verbose", action="store_true", dest="verbose", default=False)
    return parser.parse_args()
    
def main():
    
    num_cpus = multiprocessing.cpu_count()
    num_workers = num_cpus - 1
    if num_cpus > 12:
        num_workers = 12

    args = parse_arguments()

    set_seed(args.seed)
    pred_dir = join(args.outdir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = custom_transform_with_selective_augs(is_train=True, args=args)
    val_transform = custom_transform_with_selective_augs(is_train=False, args=args)       

    train_df, val_df, test_df = load_data(args, 
                                          train_size=0.9, 
                                          val_size=0.1, 
                                          random_state=42)
    
    y_train, y_val, y_test = train_df["age"], val_df["age"], test_df["age"]

    if args.train_mode == "rnc_original":
        train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, eval_test_loader = create_RnC_loaders(train_df=train_df,
                                                                                                                         val_df=val_df,
                                                                                                                         test_df=test_df,
                                                                                                                         batch_size=args.batch_size,
                                                                                                                         train_transform=train_transform,
                                                                                                                         val_transform=val_transform,
                                                                                                                         num_workers=num_workers)  

    else:
        train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, eval_test_loader = create_loaders(train_df=train_df,
                                                                                                                     val_df=val_df,
                                                                                                                     test_df=test_df,
                                                                                                                     batch_size=args.batch_size,
                                                                                                                     train_transform=train_transform,
                                                                                                                     val_transform=val_transform,
                                                                                                                     num_workers=num_workers)


    trainer = get_trainer(args, device)
    start_epoch = trainer.start_epoch
    num_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    # training loop
    best_val_mse = float("inf")
    history = {}
    history["train_loss"] = []
    history["val_loss"] = []
    history["train_mse"] = []
    history["val_mse"] = []

    print(".. Compute initial embeddings and validate")
    t0 = time.time()
    train_loss = trainer.validate_model(eval_train_loader)
    val_loss = trainer.validate_model(val_loader) 
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    Z_train, file_paths_train = trainer.compute_embeddings(eval_train_loader)
    Z_val, file_paths_val = trainer.compute_embeddings(eval_val_loader)
    train_mse, val_mse, dfp_train, dfp_val = validate_embeddings(Z_train, y_train, Z_val, y_val, args, save_coefficients=True)
    history["train_mse"].append(train_mse)
    history["val_mse"].append(val_mse)
    print("%.2f Elapsed" % (time.time() - t0))
    save_embeddings_to_csv(file_paths_train, Z_train, args.outdir, "initial_train_embeddings.csv")
    save_embeddings_to_csv(file_paths_val, Z_val, args.outdir, "initial_val_embeddings.csv")
        
    for epoch in range(start_epoch, args.num_epochs):

        # Train val loops
        print(".. Train and val loop")
        t0 = time.time()
        train_loss = trainer.train_model(args, train_loader)
        val_loss = trainer.validate_model(val_loader) 
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print("%.2f Elapsed" % (time.time() - t0))

        print(".. Compute embeddings and validate")
        t0 = time.time()
        Z_train, file_paths_train = trainer.compute_embeddings(eval_train_loader)
        Z_val, file_paths_val = trainer.compute_embeddings(eval_val_loader)
        train_mse, val_mse, dfp_train, dfp_val = validate_embeddings(Z_train, y_train,
                                                                     Z_val, y_val, 
                                                                     args)
        
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        print("%.2f Elapsed" % (time.time() - t0))

        if epoch == args.freeze_layers_until_epoch:
            
            for param in trainer.model.parameters():
                param.requires_grad = True
            trainer.lr = args.lr    
            trainer.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, 
                                                         trainer.model.parameters()), 
                                                         lr=trainer.lr, 
                                                         weight_decay=1e-5)

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_model_and_optimizer(join(args.outdir, 
                                                  f"model_epoch_%.5d.pth" % epoch))
            save_embeddings_to_csv(file_paths_train, Z_train, args.outdir, 
                                   "embeddings_and_labels_train_epoch_%.5d.csv" % epoch)
            save_embeddings_to_csv(file_paths_val, Z_val, args.outdir, 
                                   "embeddings_and_labels_val_epoch_%.5d.csv" % epoch)

        # Save best model and embeddings
        if val_mse < best_val_mse:
            print(f"Updating best_val_mse = {val_mse} at epoch {epoch}")
            best_val_mse = val_mse
            trainer.save_model_and_optimizer(join(args.outdir, "best_model.pth"))
            Z_test, file_paths_test = trainer.compute_embeddings(eval_test_loader)
            train_mse, val_mse, test_mse, dfp_train, dfp_val, dfp_test = validate_embeddings(Z_train, y_train,
                                                                                             Z_val, y_val,
                                                                                             args,
                                                                                             Z_test=Z_test,
                                                                                             y_test=y_test
                                                                                             )
            save_embeddings_to_csv(file_paths_train, Z_train, args.outdir, "best_train_embeddings.csv")
            save_embeddings_to_csv(file_paths_val, Z_val, args.outdir, "best_val_embeddings.csv")
            save_embeddings_to_csv(file_paths_test, Z_test, args.outdir, "best_test_embeddings.csv")
            print("The test_mse at the point of minimal val_mse: ", test_mse)
            with open(join(args.outdir, "test_mse_for_the_lowest_val_mse.txt"), "w") as f:
                f.write(str(test_mse))

        # Export train predictions
        outfile = join(pred_dir, "train_predictions_epoch%.5d.csv" % epoch)
        dfp_train.to_csv(outfile, index=None)

        # Export val predictions
        outfile = join(pred_dir, "val_predictions_epoch%.5d.csv" % epoch)
        dfp_val.to_csv(outfile, index=None)

        # Export training history
        if history["train_loss"]:
            outfile = join(args.outdir, "train_val_history.csv")
            pd.DataFrame(history).to_csv(outfile, index=None)
            print(f"Epoch {epoch+1}/{args.num_epochs}, Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Train MSE: {train_mse:.4f}, Validation MSE: {val_mse:.4f}")
        else:
            print("train_loss is empty. Nothing to save.")
        
if __name__ == "__main__":
    main()
