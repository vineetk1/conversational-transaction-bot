The DSTC2 dataset is about a Restaurant Reservation System, where a User
connects with this System and interacts with a Bot to make a reservation for a
table at a restaurant in a city.
The DSTC2 dataset is converted to the default format, which consists of a list
of dictionaries, where each dictionary consists of a dialog. Following example
is from a third dictionary in the dialog list, *dlgs_lst[2]*

```
{
    'persona': [],
    'user': [
        '<SILENCE>',
        'im looking for a cheap restaurant that serves tuscan food',
        '<SILENCE>',
        'im looking for a cheap restaurant that serves italian food',
        'any part of town', '<SILENCE>',
        'may i have the address and phone number please',
        'and the phone number please', 'thank you good bye'
    ],
    'bot': [
        'Hello, welcome to the Cambridge restaurant system . You can ask for
         restaurants by area, price range or food type . How may I help you ?',
        'api_call tuscan R_location cheap',
        'Sorry there is no tuscan restaurant in the cheap price range',
        'What part of town do you have in mind?',
        'api_call italian R_location cheap',
        'pizza_hut_city_centre is a nice restaurant in the centre of town
         serving italian food',
        'Sure , pizza_hut_city_centre is on pizza_hut_city_centre_address',
        'The phone number of pizza_hut_city_centre is pizza_hut_city_centre_phone',
        'you are welcome'
    ],
    'bot_idx': [1, 4],
    'api_call_result': [
     ['api_call no result'],
     [
         'zizzi_cambridge R_post_code zizzi_cambridge_post_code',
         'zizzi_cambridge R_cuisine italian',
         'zizzi_cambridge R_location centre',
         'zizzi_cambridge R_phone zizzi_cambridge_phone',
         'zizzi_cambridge R_address zizzi_cambridge_address',
         'zizzi_cambridge R_price cheap', 'zizzi_cambridge R_rating 4',
         'ask R_post_code ask_post_code', 'ask R_cuisine italian',
         'ask R_location centre', 'ask R_phone ask_phone',
         'ask R_address ask_address', 'ask R_price cheap', 'ask R_rating 2',
         'da_vinci_pizzeria R_post_code da_vinci_pizzeria_post_code',
         'da_vinci_pizzeria R_cuisine italian',
         'da_vinci_pizzeria R_location north',
         'da_vinci_pizzeria R_phone da_vinci_pizzeria_phone',
         'da_vinci_pizzeria R_address da_vinci_pizzeria_address',
         'da_vinci_pizzeria R_price cheap', 'da_vinci_pizzeria R_rating 3',
         'la_margherita R_post_code la_margherita_post_code',
         'la_margherita R_cuisine italian', 'la_margherita R_location west',
         'la_margherita R_phone la_margherita_phone',
         'la_margherita R_address la_margherita_address',
         'la_margherita R_price cheap', 'la_margherita R_rating 2',
         'pizza_hut_city_centre R_post_code pizza_hut_city_centre_post_code',
         'pizza_hut_city_centre R_cuisine italian',
         'pizza_hut_city_centre R_location centre',
         'pizza_hut_city_centre R_phone pizza_hut_city_centre_phone',
         'pizza_hut_city_centre R_address pizza_hut_city_centre_address',
         'pizza_hut_city_centre R_price cheap',
         'pizza_hut_city_centre R_rating 10'
     ]
    ]
}
```

In the above dictionary, "persona" has a list of strings where each string consists of the personality of the bot. In this example, "persona" is an empty list.   
* A User connects with a Bot and sends a message at dlgs_lst[2]['user'][0].
