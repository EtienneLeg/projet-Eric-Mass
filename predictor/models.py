from django.db import models

class ProfilAdverse(models.Model):
    TAILLE_CHOICES = [
        ('TPE', 'TPE (< 10 rep.)'),
        ('PME', 'PME'),
        ('ETI', 'ETI'),
        ('GE', 'Grand Groupe'),
    ]
    
    secteur = models.CharField(max_length=100)
    taille = models.CharField(max_length=3, choices=TAILLE_CHOICES)
    type_interlocuteur = models.CharField(max_length=50, help_text="ex: CEO, Acheteur, Technique")

    def __str__(self):
        return f"{self.secteur} - {self.get_taille_display()}"

class Argument(models.Model):
    nom = models.CharField(max_length=150, unique=True)
    famille = models.CharField(max_length=50, help_text="ex: Prix, Technique, Partenariat")

    def __str__(self):
        return self.nom

class Negociation(models.Model):
    profil = models.ForeignKey(ProfilAdverse, on_delete=models.CASCADE)
    arguments_utilises = models.ManyToManyField(Argument)
    
    pourcentage_obtenu = models.FloatField(help_text="Résultat entre 0 et 100")
    date_creation = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Négo {self.profil.secteur} - {self.pourcentage_obtenu}%"