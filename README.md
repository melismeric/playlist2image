# playlist2image

# Weekly Progress

## May 2022
- First research
- Choosing AI text to image model
- Spotify api research
## June 2022
- VQGAN-CLIP
- mood color relation
## July 2022
## August 2022
- First Notebook 
## September 2022
- custom training
- Text Generation
```python
for ind in songData.index:
  if (abstract):
   txt = "abstract " + style + " " + medium + " of " + songData['name'][ind] + ": 2" 
  else:
    txt = style + " " + medium + " of " + songData['name'][ind] + ": 2"
  
  # color 
  txt = txt + " | " + songData['Color'][ind] + " " + songData['Mood Color'][ind] + ": 0.9"

  # mood
  txt = txt + " | " + songData['nn_preds'][ind] + ": 1.8 " 
  # submood russel based mood map
  if songData['Russel Mood'][ind] not in songData['Last.fm Tag Mood'][ind]:
    txt = txt + " | " + songData['Russel Mood'][ind] + ": 0.8 "
  # last.fm 
  if songData['Last.fm Tag Mood'][ind]:
    txt = txt + " | " + songData['Last.fm Tag Mood'][ind] + ": 0.8 "

  # lighting 
  txt = txt + " | " + lighting + ": 0.9" 

  #additional
  if (songData['liveness'][ind] > 0.6):
    txt = txt +  " | concert poster: 0.9"

  if (songData['danceability'][ind] > 0.8):
    txt = txt +  " | psychedelic: 0.9"
  
  text_prompt.append(txt)
 ```

## October 2022
- mood names modification
- NN model
- Song by song image
- last.fm tags
- Russel mood chart

## November 2022
