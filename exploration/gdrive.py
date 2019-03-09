def getDrive():
      # Install the PyDrive wrapper & import libraries.
  # This only needs to be done once in a notebook.
  !pip install -U -q PyDrive
  from pydrive.auth import GoogleAuth
  from pydrive.drive import GoogleDrive
  from google.colab import auth
  from oauth2client.client import GoogleCredentials

  # Authenticate and create the PyDrive client.
  # This only needs to be done once in a notebook.
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  return GoogleDrive(gauth)

def loadFile(parentId, name):
  drive = getDrive()
  return drive.ListFile({'q': "'{}' in parents and name contains '{}'".format(parentId, name)}).GetList()
  
def driveSave(contentString):
  folder_id = "1czRhuoUEDIYo6JaqLSkuCjwdbUw6zbDW"
  
  drive = getDrive()

  # Create & upload a text file.
  uploaded = drive.CreateFile({'title': 'Sample file.txt'})
  uploaded.SetContentString(contentString)
  uploaded.Upload()
  print('Uploaded file with ID {}'.format(uploaded.get('id')))

#driveSave('Sample upload file content')

## Test 
import json

def graphListsGenerator(): 
  drive= getDrive()
  files = drive.ListFile({'q': "'{}' in parents".format("1s5tKKGpwkL7jSCqD96oiMMEnxeG0SIxN")})
  files = files.GetList()
  for file in files:
    print("Reading file {}".format(file["title"]))
    yield file.GetContentString()
    
def graphGenerator():
  glg = graphListsGenerator()
  for gl in glg:
    for g6 in gl.split("\n"):
      g = nx.from_graph6_bytes(g6)#str.encode(g6))
      yield g
    
print(graphGenerator().next())
