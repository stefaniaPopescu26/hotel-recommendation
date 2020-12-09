  model_json = open('./resources/model/model.json', 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('./resources/model/model_weights.h5')
    predictions = model.predict(np.array(skeleton_images))
    activity = decode_activity_classes[np.argmax(predictions[:])]
    activity_data = { "name": activity }
    r = requests.post(BASE_URL + 'activity', headers = HEADERS, data = json.dumps(activity_data))
    print(activity_data)
