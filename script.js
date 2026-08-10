function predict(){

let amount=document.getElementById("amount").value;
let location=document.getElementById("location").value;
let hour=document.getElementById("hour").value;

fetch("/predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({
amount:amount,
location:location,
hour:hour
})

})

.then(response=>response.json())

.then(data=>{

document.getElementById("result").innerHTML=data.prediction;

});

}