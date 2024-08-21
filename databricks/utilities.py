# Databricks notebook source

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    f"{storage_account_key}"
)
