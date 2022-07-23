"""
A Python script for downloading all files under a folder in Google Drive.
Downloaded files will be saved at the current working directory.

This script uses the official Google Drive API (https://developers.google.com/drive).
As the examples in the official doc are not very clear to me,
so I thought sharing this script would be helpful for someone.

To use this script, you should first follow the instruction 
in Quickstart section in the official doc (https://developers.google.com/drive/api/v3/quickstart/python):
- Enable Google Drive API 
- Download `credential.json`
- Install dependencies


Notes:
- This script will only work on a local environment, 
  i.e. you can't run this on a remote machine
  because of the authentication process of Google.
- This script only downloads binary files not google docs or spreadsheets.


Author: Sangwoong Yoon (https://github.com/swyoon/)
"""
import io
import pickle
import os.path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload 
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

'''Configuration'''
# ID of the folder to be downloaded.
# ID can be obtained from the URL of the folder
FOLDER_ID = '13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D'  # an example folder



# If modifying these scopes, delete the file token.pickle.
# SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def main():
    """Download all files in the specified folder in Google Drive."""
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    page_token = None
    while True:
        # Call the Drive v3 API
        results = service.files().list(
                q=f"'{FOLDER_ID}' in parents",
                pageSize=10, fields="nextPageToken, files(id, name)",
                pageToken=page_token).execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
        else:
            for item in items:
                print(u'{0} ({1})'.format(item['name'], item['id']))


                file_id = item['id']
                request = service.files().get_media(fileId=file_id)

                with open(item['name'], 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        print("Download %d%%." % int(status.progress() * 100))

        page_token = results.get('nextPageToken', None)
        if page_token is None:
            break

if __name__ == '__main__':
    main()
