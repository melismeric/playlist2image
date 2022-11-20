# Pixtape Tool Creation and Evaluation: How well can VQGAN-CLIP visualize the mood of a Spotify playlists?


# Weekly Progress

## May 2022

# Comparing Text To Image Generation Models
In the beginning of the project I decided to use text to image generation techniques to visualize Spotify playlists. I started my project by
comparing the text-to-image generation models. Below is a list of the notebooks I tried before selecting VQGAN-CLIP.

**Notebooks:**

- **DALL-E Mini** [https://huggingface.co/spaces/dalle-mini/dalle-mini](https://huggingface.co/spaces/dalle-mini/dalle-mini) 
- **The Big Sleep: BigGANxCLIP:** [https://colab.research.google.com/github/levindabhi/CLIP-Notebooks/blob/main/The_Big_Sleep_BigGANxCLIP.ipynb](https://colab.research.google.com/github/levindabhi/CLIP-Notebooks/blob/main/The_Big_Sleep_BigGANxCLIP.ipynb) 
- **VQGAN+CLIP:** [https://colab.research.google.com/github/justinjohn0306/VQGAN-CLIP/blob/main/VQGAN%2BCLIP(Updated).ipynb#scrollTo=VA1PHoJrRiK9](https://colab.research.google.com/github/justinjohn0306/VQGAN-CLIP/blob/main/VQGAN%2BCLIP(Updated).ipynb#scrollTo=VA1PHoJrRiK9) 
- **Disco Diffusion:** [https://colab.research.google.com/drive/1sHfRn5Y0YKYKi1k-ifUSBFRNJ8_1sa39](https://colab.research.google.com/drive/1sHfRn5Y0YKYKi1k-ifUSBFRNJ8_1sa39)
- **CLIP and StyleGAN:** [https://colab.research.google.com/github/orpatashnik/StyleCLIP/blob/main/notebooks/optimization_playground.ipynb](https://colab.research.google.com/github/orpatashnik/StyleCLIP/blob/main/notebooks/optimization_playground.ipynb)
- **ruDall-e** [https://github.com/ai-forever/ru-dalle](https://github.com/ai-forever/ru-dalle)
- [https://colab.research.google.com/drive/1sc-Qa7feIatfWkhHWFu01pzeE2AVinmn?usp=sharing#scrollTo=Pf8a78a2WKoU](https://colab.research.google.com/drive/1sc-Qa7feIatfWkhHWFu01pzeE2AVinmn?usp=sharing#scrollTo=Pf8a78a2WKoU)

I tested these models with song names to have an ide of their style. 
Here are some images from DALLE and Big Sleep.
![image](https://user-images.githubusercontent.com/37816087/202878226-abaabe6a-1e56-463f-b7cc-2f9b9150a135.png)
![image](https://user-images.githubusercontent.com/37816087/202878241-3c2db503-eea9-45e8-b51b-9acceb6527a2.png)


VQGA-CLIP
![image](https://user-images.githubusercontent.com/37816087/202878280-68dddc66-8785-4e7c-baea-2da6c0468c59.png)

I decided to use VQGAN-CLIP as it depicts the images better given a style and color, like " Fauvist oil on canvas, blue shades"

# Spotify API research
I decided to use Spotify API to get the audio features. Hence I did some research to learn the limitations of the data source and available audio features.
Here are the audio features and their descriptions I will use in my project to predict the moods, and colors of the playlists.
* Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable<img width="683" alt="Screen Shot 2022-11-20 at 01 40 06" src="https://user-images.githubusercontent.com/37816087/202878681-0a8eaedf-7677-4430-86c2-9d3b72c140f6.png">

* Liveness Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
* Loudness The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
* Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
* Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
* Valence A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
* Acousticness A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.

## June 2022

# VQGAN+CLIP EXPERIMENTS
In the first weeks of June I did some tests with VQGAN-CLIP with multiple playlists to see how the images change according to the playlist. 
I used single text prompt including song names in a playlist and a style keyword . Here are some examples.

Text: "'Sun', 'Kiwi', 'Take on Me', 'Golden', 'Girl Crush, London', 'Canyon Moon', 'Adore You', 'Me Without You', 'Watermelon Sugar', 'High', 'Cherry', 'Kill My Mind', 'Summer Is for Falling in Love', 'Homage', 'Lovesong', "Itchin'", "abstract cubism"
Output:
![image](https://user-images.githubusercontent.com/37816087/202878590-ef4f3dad-d964-4fe5-a526-11a0d1421524.png)

Then I decided to change the text format and used | for seperation and used weighting for each word, which helped me to get better results.
Text: Our Song Island In The Sun Sun Sunflower, Vol. 6 Only Angel Kiwi Take on Me Golden Girl Crush Canyon Moon Back to You Lights Up good 4 u Adore You Me Without You Watermelon Sugar High Cherry Kill My Mind Summer Is for Falling in Love:1 | sunrays shine upon it: 0.9
Output: ![image](https://user-images.githubusercontent.com/37816087/202878639-ca98093a-053e-465c-b966-a269619c83b7.png)

### Trying Different Keywords for style
From this source(https://imgur.com/a/SALxbQm) I found out that VQGAN-CLIP is quitee good depicting different styles for an image.

Hence for the same playlist I tested keywords like, "abstract cubist","melancholic holographic","concert poster" etc.
<img width="681" alt="Screen Shot 2022-11-20 at 01 39 16" src="https://user-images.githubusercontent.com/37816087/202878669-ee24ddba-fe8d-4191-8f6a-7055f4732245.png">
<img width="674" alt="Screen Shot 2022-11-20 at 01 42 03" src="https://user-images.githubusercontent.com/37816087/202878735-d08a38f7-4777-4cce-ac42-4c578f480df9.png">

### Comparing Images of different Iterations of VQGAN-CLIP

<img width="500" alt="Screen Shot 2022-11-20 at 01 46 23" src="https://user-images.githubusercontent.com/37816087/202878835-bc557e06-eee8-4b98-bcff-a5caa135d615.png">
After seeing the results of different playlists, I decided to use 200-250 iterations of the images instead of more than 800 iterations. The images get more chaotic and worse in the style afteer too many iterations.

Here are the collected outputs of the VQGAN-CLIP experiments I did: https://balanced-romano-f52.notion.site/VQGAN-CLIP-EXPERIMENTS-a0b93a406b684e3d8d4a1e3330dd303e

### Generating Images with muultiple text prompts instead of a single one:
After seeing the images of playlists using a text prompt inlcuding all of the song names I found the obtained imag not very compelling.
Hence I decided to generate images song by song using a text array including different texts for each song. I also decided to make a video 
from the generated images to see. the change from one song to another.
<img width="710" alt="Screen Shot 2022-11-20 at 01 54 41" src="https://user-images.githubusercontent.com/37816087/202879004-efb3e12e-df24-4c82-9f81-1367078dd042.png">

At the end of June I decided to use multiple text prompts including songs of the playlist with a style keyword which will be same for evrry text in the
array to have a compact style for a playlist.

# Mood-Color relation research

## July 2022

### VQGAN-CLIP Experiments with different keywords
I testes different kins of text formats to find the best one for my project.

Text formats:
1- Song names + color name + image type(abstract fauvism)
2- Song names + mood name (melancholic, positive) + image type(abstract fauvism)

### The Keywords:
I also created a table to test different keywords and contexts
<img width="600" alt="Screen Shot 2022-11-20 at 01 59 30" src="https://user-images.githubusercontent.com/37816087/202879113-c5be3f33-f4f0-46bd-a733-36cae723abb7.png">

## August 2022
- First Notebook 
## September 2022
Collection of playlist results:
[https://www.notion.so/Spotify-Playlist-Results-c05c5d988e1849088912e011c2d17e1a](https://balanced-romano-f52.notion.site/Spotify-Playlist-Results-c05c5d988e1849088912e011c2d17e1a)
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
