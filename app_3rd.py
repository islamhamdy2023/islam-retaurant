import streamlit as st
import pandas as pd
import joblib

Inputs = joblib.load(r"F:\data science\final project\islam taha\restaurant\Inputs.pkl")
Model = joblib.load(r"F:\data science\final project\islam taha\restaurant\Model.pkl")

def prediction(online_order,book_table,votes,location,
       approx_cost_for_two_people,listed_in_type):
    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0,"online_order"] = online_order
    test_df.at[0,"book_table"] = book_table
    test_df.at[0,"votes"] = votes
    test_df.at[0,"location"] = location
    test_df.at[0,'approx_cost_for_two_people'] = approx_cost_for_two_people
    test_df.at[0,"listed_in_type"] = listed_in_type
    st.dataframe(test_df)
    result = Model.predict(test_df)[0]
    return result


    
def main():
    st.image("https://www.condor.cl/wp-content/uploads/2022/10/4504_P-8-9-10_2.jpg")
    st.title("Bangolre Resturants")
    online_order = st.selectbox("online" , ['Yes', 'No'])
    book_table = st.selectbox("book_table" , ['Yes', 'No'])
    votes = st.slider("votes" , min_value= 0 , max_value=16832 , value=0,step=1)
    location = st.selectbox("location" ,['Banashankari', 'Basavanagudi', 'Mysore Road', 'Jayanagar',
       'Kumaraswamy Layout', 'Rajarajeshwari Nagar', 'Vijay Nagar',
       'Uttarahalli', 'JP Nagar', 'South Bangalore', 'City Market',
       'Nagarbhavi', 'Bannerghatta Road', 'BTM', 'Kanakapura Road',
       'Bommanahalli', 'CV Raman Nagar', 'Electronic City', 'HSR',
       'Marathahalli', 'Wilson Garden', 'Shanti Nagar',
       'Koramangala 5th Block', 'Koramangala 8th Block', 'Richmond Road',
       'Koramangala 7th Block', 'Jalahalli', 'Koramangala 4th Block',
       'Bellandur', 'Sarjapur Road', 'Whitefield', 'East Bangalore',
       'Old Airport Road', 'Indiranagar', 'Koramangala 1st Block',
       'Frazer Town', 'RT Nagar', 'MG Road', 'Brigade Road',
       'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road',
       'Shivajinagar', 'Infantry Road', 'St. Marks Road',
       'Cunningham Road', 'Race Course Road', 'Commercial Street',
       'Vasanth Nagar', 'HBR Layout', 'Domlur', 'Ejipura',
       'Jeevan Bhima Nagar', 'Old Madras Road', 'Malleshwaram',
       'Seshadripuram', 'Kammanahalli', 'Koramangala 6th Block',
       'Majestic', 'Langford Town', 'Central Bangalore', 'Sanjay Nagar',
       'Brookefield', 'ITPL Main Road, Whitefield',
       'Varthur Main Road, Whitefield', 'KR Puram',
       'Koramangala 2nd Block', 'Koramangala 3rd Block', 'Koramangala',
       'Hosur Road', 'Rajajinagar', 'Banaswadi', 'North Bangalore',
       'Nagawara', 'Hennur', 'Kalyan Nagar', 'New BEL Road', 'Jakkur',
       'Rammurthy Nagar', 'Thippasandra', 'Kaggadasapura', 'Hebbal',
       'Kengeri', 'Sankey Road', 'Sadashiv Nagar', 'Basaveshwara Nagar',
       'Yeshwantpur', 'West Bangalore', 'Magadi Road', 'Yelahanka',
       'Sahakara Nagar', 'Peenya'] )
    listed_in_type = st.selectbox("listed_in_type" , ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
       'Drinks & nightlife', 'Pubs and bars'])
    approx_cost_for_two_people= st.slider("approx_cost_for_two_people)" , min_value=40, max_value=6000, value=0, step=1)
    if st.button("predict"):
        result = prediction(online_order,book_table,votes,location,
       approx_cost_for_two_people,listed_in_type)
        label = ["NOT GOOD ENOUGH" , "success"]
        st.text(f"The Resturant will be {label[result]}")
        
if __name__ == '__main__':
    main()    
    
