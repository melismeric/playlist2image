# Pixtape Tool Creation and Evaluation: How well can VQGAN-CLIP visualize the mood of a Spotify playlists?
Video: https://youtu.be/gkJfaY45cWo
Notebook: https://colab.research.google.com/drive/11b-NGMrYnOxRLsEd-ts6hxk0vrSbeS96?usp=sharing
<img width="920" alt="Screen Shot 2022-11-21 at 01 00 24" src="https://user-images.githubusercontent.com/37816087/202937201-ccf51cc8-cd78-47fd-b504-a06ce43a3ad1.png">

# Weekly Progress

## May 2022

### Comparing Text To Image Generation Models
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
Here are some images from DALL-E and Big Sleep.

DALL-E:

<img width="200" alt="Screen Shot 2022-11-20 at 01 39 16" src="https://user-images.githubusercontent.com/37816087/202878226-abaabe6a-1e56-463f-b7cc-2f9b9150a135.png">

Big Sleep:

<img width="210" alt="Screen Shot 2022-11-20 at 01 39 16" src="https://user-images.githubusercontent.com/37816087/202878241-3c2db503-eea9-45e8-b51b-9acceb6527a2.png">

VQGAN-CLIP:

<img width="200" alt="Screen Shot 2022-11-20 at 01 39 16" src="https://user-images.githubusercontent.com/37816087/202878280-68dddc66-8785-4e7c-baea-2da6c0468c59.png">


I decided to use VQGAN-CLIP as it depicts the images better with given a style and color, like " Fauvist oil on canvas, blue shades"

### Spotify API research
I decided to use Spotify API to get the audio features. Hence I did some research to learn the limitations of the data source and available audio features.
Here are the audio features and their descriptions which I will use in my project to predict the moods, and colors of the playlists.
* Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable
* Liveness Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
* Loudness The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
* Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
* Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
* Valence A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
* Acousticness A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.

## June 2022

### VQGAN+CLIP EXPERIMENTS
In the first weeks of June I did some tests with VQGAN-CLIP with multiple playlists to see how the images change according to the playlist. 
I used single text prompt including song names in a playlist and a style keyword . Here are some examples.

Text: "'Sun', 'Kiwi', 'Take on Me', 'Golden', 'Girl Crush, London', 'Canyon Moon', 'Adore You', 'Me Without You', 'Watermelon Sugar', 'High', 'Cherry', 'Kill My Mind', 'Summer Is for Falling in Love', 'Homage', 'Lovesong', "Itchin'", "abstract cubism"

Output:

<img width="200" alt="Screen Shot 2022-11-20 at 01 39 16" src="https://user-images.githubusercontent.com/37816087/202878590-ef4f3dad-d964-4fe5-a526-11a0d1421524.png">


Then I decided to change the text format and used | for seperation and used weighting for each word, which helped me to get better results.

Text: Our Song Island In The Sun Sun Sunflower, Vol. 6 Only Angel Kiwi Take on Me Golden Girl Crush Canyon Moon Back to You Lights Up good 4 u Adore You Me Without You Watermelon Sugar High Cherry Kill My Mind Summer Is for Falling in Love:1 | sunrays shine upon it: 0.9

Output: 

<img width="200" alt="Screen Shot 2022-11-20 at 01 39 16" src="https://user-images.githubusercontent.com/37816087/202878639-ca98093a-053e-465c-b966-a269619c83b7.png">


### Trying Different Keywords for style
https://imgur.com/a/SALxbQm is a great source for different keywords to try VQGAN-CLIP.

Hence for the same playlist I tested keywords like, "abstract cubist","melancholic holographic","concert poster" etc.
<img width="400" alt="Screen Shot 2022-11-20 at 01 39 16" src="https://user-images.githubusercontent.com/37816087/202878669-ee24ddba-fe8d-4191-8f6a-7055f4732245.png">
<img width="400" alt="Screen Shot 2022-11-20 at 01 42 03" src="https://user-images.githubusercontent.com/37816087/202878735-d08a38f7-4777-4cce-ac42-4c578f480df9.png">

### Comparing Images of different Iterations of VQGAN-CLIP

<img width="500" alt="Screen Shot 2022-11-20 at 01 46 23" src="https://user-images.githubusercontent.com/37816087/202878835-bc557e06-eee8-4b98-bcff-a5caa135d615.png">
After seeing the results of different playlists, I decided to use 200-250 iterations of the images instead of more than 800 iterations. The images get more chaotic and worse in the style after too many iterations.

Here are the collected outputs of the VQGAN-CLIP experiments I did: https://balanced-romano-f52.notion.site/VQGAN-CLIP-EXPERIMENTS-a0b93a406b684e3d8d4a1e3330dd303e

### Generating Images with multiple text prompts instead of a single one:
After seeing the images of playlists using a text prompt inlcuding all of the song names I found the obtained image not very compelling.
Hence I decided to generate images song by song using a text array including different texts for each song. I also decided to make a video 
from the generated images to see the change from one song to another.
<img width="710" alt="Screen Shot 2022-11-20 at 01 54 41" src="https://user-images.githubusercontent.com/37816087/202879004-efb3e12e-df24-4c82-9f81-1367078dd042.png">

At the end of June, I decided to use multiple text prompts including songs of the playlist with a style keyword which will be same for every text in the
array to have a compact style for a playlist.

### Mood-Color relation research
<img width="620" alt="Screen Shot 2022-11-21 at 01 05 54" src="https://user-images.githubusercontent.com/37816087/202937592-58784044-0240-4dc8-95ca-85aef327eaa7.png">
Thayer's Mood Model (Thayer, 1989)

<img width="726" alt="Screen Shot 2022-11-21 at 01 08 16" src="https://user-images.githubusercontent.com/37816087/202937792-6a3e5974-1e66-4624-812d-38de532a0a54.png">
(N. A. Nijdam, 2009) 

Here are the collected resources I studied for mood-color relation. https://balanced-romano-f52.notion.site/Mood-Analysis-1389b66da1e942139fd8b6b43b5b881c

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

### Testing Spotify API and Color Mapping
<img width="480" alt="Screen Shot 2022-11-21 at 01 01 00" src="https://user-images.githubusercontent.com/37816087/202937243-fb33333d-d668-41d0-9ced-69dc5433be12.png">
(Bakhshizadeh et al., 2019)

https://colab.research.google.com/drive/1Il1yHKDbG27Pyng713mt6lZ2FW4kFsQt#scrollTo=eRjdb_R4wWXP

A main and mood color is constructed for each playlist by getting the average of songs’ audio features.
Audio Features for color mapping: ['danceability','energy','liveness', 'valence', 'acousticness', 'mode']

**Main Color:** (R,G,B) = (Energy, Valence, Acousticness)

**Mood Color**:

- Converts main color to HSV to add danceability and mode.
    Mode
    if mode == 1 :
      major/bright colors/ higher v values
    else:
      minor/dark colors/ lower v values

    Danceability
    High danceability = higher s values
    Low danceability = lower s values
```python
def modifyColor(d,m,r,g,b):
  (h, s, v) = rgb_to_hsv(r, g, b)
  (r, g, b) = hsv_to_rgb(h, d, (1-m)*255)
  return (round(r), round(g), round(b))
```
<img width="500" alt="Screen Shot 2022-11-21 at 01 19 54" src="https://user-images.githubusercontent.com/37816087/202938634-d4a03cf3-601b-4f7d-830c-e434c70ea913.png">
<img width="500" alt="Screen Shot 2022-11-21 at 01 20 06" src="https://user-images.githubusercontent.com/37816087/202938648-46382a02-7b2b-4e1b-9cb0-0e075afcddd7.png">

### Mood Mapping
## Mood Detection

I used Gracenote mood map created by Neo, T. (2017).

<img width="680" alt="Screen Shot 2022-11-21 at 01 24 00" src="https://user-images.githubusercontent.com/37816087/202938966-a2ce9899-08a3-4969-9cc6-a3b0b1ee804f.png">


```python
def moodDetect(mainColor):
  mood = ""
  if (mainColor == "red" or mainColor == "maroon"):
    mood = "anger"
  elif (mainColor == "purple" or mainColor == "navy"):
    mood = "euphoric"
  elif (mainColor == "pink"):
    mood = "romantic"
  elif (mainColor == "yellow" ):
    mood = "positive and sunrays shine upon it"
  elif (mainColor == "orange" ):
    mood = "poweful and dynamic and sunrays shine upon it"
  elif (mainColor == "green" or mainColor == "olive" or mainColor == "teal"):
    mood = "calm"
  elif (mainColor == "blue" or mainColor == "grey" or mainColor == "black"):
    mood = "melancholic"
  return mood


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

# better way to detect mood (https://neokt.github.io/projects/audio-music-mood-classification/)
def getMood(energy, valence):
  keywords = ["Somber", "Gritty", "Serious", "Brooding", "Aggressive", "Melancholy", "Cool", "Yearning", "Urgent", "Defiant", "Sentimental", "Sophisticated", "Sensual", "Fiery", "Energizing", "Tender", "Romantic",  "Empowering", "Stirring", "Rowdy", "Peaceful", "Easygoing", "Upbeat", "Lively", "Excited"]
  array = [(0.2, 0.2), (0.4, 0.2), (0.5, 0.2), (0.6, 0.2), (0.8, 0.2), (0.2, 0.4), (0.4, 0.4), (0.5, 0.4), (0.6, 0.4), (0.8, 0.4), (0.2, 0.5), (0.4, 0.5), (0.5, 0.5), (0.6, 0.5), (0.8, 0.5), (0.2, 0.6), (0.4, 0.6), (0.5, 0.6), (0.6, 0.6), (0.8, 0.6), (0.2, 0.8), (0.4, 0.8), (0.5, 0.8), (0.6, 0.8), (0.8, 0.8)]
  mood = keywords[closest_node((energy, valence), array)]
  return mood
```

## September 2022
In September I used new mood labels, and neural network for predicting the moods.

### Comparing Different Mood Labels

Russell Mood model has a similar taxonomy with the Thayer's mood chart with 2 dimensions of energy and valence. As Russell Mood Model is has less mood labels than Gracenote mood labels and has a circular structure I decided to compare them.

I used Last.fm user generated tags as they fit well with the Russell's Mood Model
 
<img width="500" alt="Screen Shot 2022-11-21 at 01 31 49" src="https://user-images.githubusercontent.com/37816087/202941346-e53edcf7-37b9-41ae-be34-270778518485.png">

Russell, J.A. (1980)

<img width="500" alt="Screen Shot 2022-11-21 at 01 32 08" src="https://user-images.githubusercontent.com/37816087/202941431-76dd8cf8-da64-4fd9-8272-1895dcecc97b.png">
Downie and Hu’s mood chart (2010)


Here is the comparison of images generated with Last.fm Mood Labels and Gracenote Mood Labels.
<img width="500" alt="Screen Shot 2022-11-21 at 01 24 15" src="https://user-images.githubusercontent.com/37816087/202938985-1f4911e9-c1fe-4be4-bdac-a41be2f996ac.png">

Collection of playlist results:
[https://www.notion.so/Spotify-Playlist-Results-c05c5d988e1849088912e011c2d17e1a](https://balanced-romano-f52.notion.site/Spotify-Playlist-Results-c05c5d988e1849088912e011c2d17e1a)
### Custom Training

It is also possible to train VQGAN on custom datasets. In my project, I decided to try training VQGAN with the album covers dataset (Greg, 2019). I used Taming Transformers code source (Esser et al., 2021) to do the training. However, due to the limitations in the time and sources, I wasn’t able to obtain the results that I desired. Below are some outcomes of the custom-trained VQGAN with 3 different text prompts.

Notebook For Custom Training:

https://colab.research.google.com/drive/1lrV84QRj9NJoyKLMmfeOH7gtqmkpeMEU?usp=sharing
<img width="631" alt="Screen Shot 2022-11-21 at 01 47 43" src="https://user-images.githubusercontent.com/37816087/202943622-f9b993da-94e3-4777-a015-94a59d35c896.png">

Detailed notes:
https://balanced-romano-f52.notion.site/CUSTOM-TRAINING-3783d41a3c854f60b2f405fc8952378d

### Text Generation
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
### Using Neural Networks for Mood Prediction
Here is the notebook I tested Logistic Regression and Neural Networks.
https://colab.research.google.com/drive/1tkz-RlbT3f5potsbTMm56hkgG2n1DR8A?usp=sharing

**Main Mood using Neural Networks**

Along with the mood map I used neural networks to predict the mood of each song. 

**Collecting Data** 

To train the neural network I used Spotify mood playlists like [Ting Neo’s,](https://neokt.github.io/projects/audio-music-mood-classification/) [Cristóbal Veas’s](https://towardsdatascience.com/predicting-the-music-mood-of-a-song-with-deep-learning-c3ac2b45229e) and K [Singh’s](https://medium.com/codex/music-mood-classification-using-neural-networks-and-spotifys-web-api-d73b391044a4) respective works on Music Mood Classification. Spotify creates playlists for moods considering audio features of songs. The data is obtained from Spotify using the `spotipy`
Python library. I created a dataset with 5 mood class dataset as I found the accuracy is better with the 5 mood class dataset than higher numbered mood classes datasets. I selected Spotify mood playlists by first testing the code with the moods : 'Energetic', 'Relaxing', 'Dark', 'Sad', 'Happy' , 'Focused', 'Romantic' however the accuracy was 0.4. Then reduced the mood classes and got the best accuracy as 0.7 by selecting the mood classes as 'Energetic', 'Relaxing', 'Sad', 'Happy' , 'Focused', each class having 7 playlists. In total the dataset to train the model had 3100 songs. 

**Preparing Train and Test Datasets**

Then I split the data into **training and test sets**. **The training set has 2077 tracks and the test set has 1023 tracks.**

```python
trainx, testx, trainy, testy = train_test_split(data, moods, test_size = 0.33, random_state = 42, stratify=moods)
```

Using the `stratify` parameter ensures that the class distribution for our train and test data is the same.

**Logistic Regression**

Firstly Logistic Regression model is trained and got 67% accuracy using cross-validation. The training set is used to **cross-validate the model and test set is used to** evaluate the model after it is optimized in cross-validation. 

We can use Logistic Regression to look at the **importance given of audio features to each mood class.** This can be obtained by calculating the Euler number to the power of the coefficients of the logistic regression model. The following table shows the audio feature with the highest importance for each mood.

```python
Energetic            loudness
Focused      instrumentalness
Happy                 valence
Relaxing         acousticness
Sad                  loudness
```

**Neural Networks**

After getting the accuracy of Logistic Regression I trained a NeuraL Networks model to see if I will be able to get a better accuracy. A neural network is consisted of input, hidden and output layers. **The number of neurons in the hidden layer can be taken as the average of the number of units in the input and output layers**.(resource) Our input has 10 classes, and the output layer has 5. Hence I started with 8 units in the hidden layer and got a cross-validation accuracy of **69%**, higher than the logistic regression classifier.

The **hyperparameters** “alpha” can be optimized denoting the amount of regularization and the number of neurons in the only hidden layer of our NN. Using 10 neurons in the hidden layer and `alpha` as 0.1, the model gets a cross-validation accuracy of 71**%**.

**Analyzing the Model** 

After deciding to use NN2 model, I run some analyzes on the model. Table 2. shows the learning curve for thee model. Learning curve shows helps to determine if adding more data will help in improving the accuracy. Based on the learning curve, we can conclude that the training score and validations score curves are about to be converged, hence adding more data would not help much. In the future work more data can be added to get the best accuracy, however for this phase of the project it is decided to use a dataset with 3100 songs.

<img width="628" alt="Screen Shot 2022-11-21 at 01 50 37" src="https://user-images.githubusercontent.com/37816087/202943895-67766950-c320-4e32-9d3c-2891cb60142f.png">

Table 2 Learning Curve

**Error Metrics**

In addition to accuracy, I found it useful to discuss the error metrics represented as a confusion matrix as this is a multi-label classification problem.

<img width="422" alt="Screen Shot 2022-11-21 at 01 50 55" src="https://user-images.githubusercontent.com/37816087/202943920-28944631-d049-4004-af19-5c9bc86e36c9.png">

Table 3 Confusion Matrix

Based on the confusion matrix the highest number of incorrect classification seemed to be  Relaxing tracks being classified as Sad. These incorrect classifications can be fixed in the future work.

**Using the Model to Predict the Mood for Each Song**

After getting a 71% accuracy with the neural network model I added the model to the Colab Notebook to classify given playlists’ songs. The predicted mood for each song will be the main mood to add the text prompt.


## November 2022
Final version of the Project Pipeline:
<img width="500" alt="Screen Shot 2022-11-21 at 00 58 27" src="https://user-images.githubusercontent.com/37816087/202937075-86c2d863-df9f-4c52-98bb-290abdaeb789.png">

Text Format:
<img width="851" alt="Screen Shot 2022-11-21 at 00 59 08" src="https://user-images.githubusercontent.com/37816087/202937109-96d00fe2-e5ac-4ba3-9e11-74bd9482fe60.png">

# Resources:
1. Bakhshizadeh, M. et al. (2019) “Automated mood based music playlist generation by clustering the audio features,” 2019 9th International Conference on Computer and Knowledge Engineering (ICCKE) [Preprint]. Available at: https://doi.org/10.1109/iccke48569.2019.8965190.
2. Hu, X & Stephen Downie, J 2010, When lyrics outperform audio for music mood classification: A feature analysis. in Proceedings of the 11th International Society for Music Information Retrieval Conference, ISMIR 2010.  Proceedings of the 11th International Society for Music Information Retrieval Conference, ISMIR 2010, pp. 619-624, 11th International Society for Music Information Retrieval Conference, ISMIR 2010, Utrecht, Netherlands, 8/9/10.
3. N. A. Nijdam,(2009) "Mapping emotion to color," Book mapping emotion to 
color, pp. 2-9,
3. Neo, T. (2017) Audio Music Mood Classification, Tinkering with Data. Available at: https://neokt.github.io/projects/audio-music-mood-classification/ (Accessed: November 20, 2022).
4. Thayer, R. E. (1989). “The Biopsychology of Mood and Arousal”, New York: Oxford University Press.
5. Russell, J.A. (1980) “A circumplex model of affect.,” Journal of Personality and Social Psychology, 39(6), pp. 1161–1178. Available at: https://doi.org/10.1037/h0077714.
6. Api docs (no date) Last.fm. Available at: https://www.last.fm/api (Accessed: November 20, 2022). 
