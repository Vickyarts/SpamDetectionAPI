from django.db import models




class User(models.Model):
    name = models.CharField(max_length=100)
    key = models.CharField(max_length=24)
    created_at = models.DateTimeField()


class Admin(models.Model):
    name = models.CharField(max_length=100)
    key = models.CharField(max_length=36)
    created_at = models.DateTimeField()


class Dataset(models.Model):
    dataset_id = models.CharField(max_length=24)
    row_count = models.IntegerField()
    value_ratio = models.CharField(max_length=10, help_text="Ratio: Ham:Spam")
    created_by = models.IntegerField()
    created_at = models.DateTimeField()


class Models(models.Model):
    model_id = models.CharField(max_length=24)
    dataset_id = models.CharField(max_length=24)
    naive_accuracy = models.FloatField()
    naive_f1_score = models.FloatField()
    network_bce_loss = models.FloatField()
    created_by = models.IntegerField()
    created_at = models.DateTimeField()


class TrainingJob(models.Model):
    job_id = models.CharField(max_length=20)
    dataset_id = models.CharField(max_length=24)
    status = models.CharField(max_length=10)
    created_by = models.IntegerField()
    created_at = models.DateTimeField()


class Deployment(models.Model):
    model_id = models.CharField(max_length=24)
    type = models.CharField(max_length=15)
    created_by = models.IntegerField()
    created_at = models.DateTimeField()
