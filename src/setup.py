import os
import dataloader
import dataprocessor224

#Pfade, wo die Config-Datei des UAVVaste-Projekts ist
anns_path = os.path.abspath(os.path.join('config', 'annotations', 'annotations.json'))
tvt_path = os.path.abspath(os.path.join( 'config', 'annotations', 'train_val_test_distribution_file.json'))

#Daten herunterladen und in 224-pixel aufteilen
def __main__():
    data = dataloader.DataLoader(anns_path, tvt_path)
    dataprocessor224.image_transformer(data)
    return

if __name__ == "__main__":
    __main__()
