# GrammarChecker (Frasakan)
Merupakan proyek Torche Education Frasakan, dimana proyek ini dilatarbelakangi oleh kebutuhan mahasiswa/siswa dalam melakukan penulisan formal ilmiah dalam Bahasa Indonesia. Adapun dalam implementasinya, digunakannya T5 model yang memiliki arsitektur sebagai berikut:

![Architecture-of-the-T5-model](https://github.com/user-attachments/assets/b6ac48ed-9f84-464a-9907-fec6dfcdbee9)

Sumber: Wang, Mingye & Xie, Pan & Du, Yao & Hu, Xiaohui. (2023). T5-Based Model for Abstractive Summarization: A Semi-Supervised Learning Approach with Consistency Loss Functions. Applied Sciences. 13. 7111. 10.3390/app13127111. 


## Dataset
Dataset yang digunakan diambil melalui beberapa sumber Github dan situs pers (Liputan, Kompas, dll) yang dilakukan augmentasi substitusi menggunakan kata slang, penghapusan kata, penambahan kata redundan, dan pengubahan posisi kata dalam suatu kalimat. Adapun augmentasi dilakukan sebesar 10%-50% dari total kata yang ada pada masing-masing kalimat. Dataset ini terdiri kolom kalimat informal dan kolom kalimat formal secara berpasangan.

[Indonesian Slang Words](https://github.com/louisowen6/NLP_bahasa_resources) | [Dataset](https://drive.google.com/drive/folders/1OyTz6T2lDIvF9aEzF0Jcm1YsZyr21ojK?usp=sharing)

## UI
Adapun UI yang digunakan merupakan konsep sederhana yang dibuat menggunakan framework Streamlit dengan pemilihan model checkpoint dari hasil pelatihan T5 pada dataset yang terlampir. Selanjutnya, pengguna akan memberikan input teks informal pada lalu tekan convert.

![Dashboard](https://github.com/user-attachments/assets/03391742-c947-4824-aead-92ade0a50de4)
