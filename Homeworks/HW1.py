#GLOBAL AI-MAKİNE ÖĞRENMESİ ÖDEV 1
#1.	Tehlikeli proseslerde insan yoğunluğunun azaltılması veya tamamen otomasyona dayalı hale getirilmesi, hali hazırdaki proseslerin hızlandırılması, verimliliğinin artırılması, iyileştirilmesi, gelecek proseslerle ilgili tahminlerin yapılması, akıllı şehir, şebeke ve trafik  veya güvenlikte kullanılan sürü drone gibi büyük verilere dayalı proseslerin yürütülmesi için bu sistemlere yapılması istenilen işlemlerin öğretilmesi ve öğrenilen işlemlerin giderek iyileştirilmesi gerekmektedir. Bu sistemlerin genel çatısını Yapay zeka  (AI) oluştururken temel yapısını Makine Öğrenmesi (ML) oluşturmaktadır. Bu metot danışmanlı (supervised) ve danışmansız (unsupervised) şeklinde bir çıktıya bağlı olarak veya olmaksızın çalışma prensibine dayanmaktadır. Ayrıca takviyeli (reinforcement) şeklinde ek bir yöntem de mevcuttur. Bununla alakalı birçok yöntem geliştirilmiş olup; en güncelini derin öğrenme yöntemi oluşturmaktadır.
#2.	
#3.	Makine öğrenmesi giriş verisinin bir çıktıya bağlı olarak veya olmaksızın eğitimine göre danışmanlı (supervised) ve danışmansız (unsupervised) öğrenme şeklinde çalışma prensibine dayanmaktadır. Denetimli öğrenme algoritması için sınıflandırma ve regresyon yöntemleri mevcutken; denetimsiz için kümeleme yöntemi kullanılmaktadır.
#Denetimli makine öğrenmesi algoritmaları destek vektör makinesi(SVM) ,  karar ağaçları , k-en yakın komşu  ve yapay sinir ağları  vs. içerir.

#K- ortalama kümeleme : K, her yineleme için en yüksek değeri bulmanıza yardımcı olan yinelemeli bir kümeleme algoritması anlamına gelir. Başlangıçta, istenilen sayıda küme seçilir. Bu kümeleme yönteminde, veri noktalarını k gruplarına kümelemeniz gerekir. Daha büyük bir k, aynı şekilde daha fazla ayrıntıya sahip daha küçük gruplar anlamına gelir.

#Hiyerarşik kümeleme : Veri noktalarınızı üst ve alt kümeler halinde kümeler. Müşterilerinizi daha genç ve daha büyük yaşlara bölebilir ve ardından bu grupların her birini kendi bireysel kümelerine de bölebilirsiniz.

#Olasılıksal kümeleme :  Veri noktalarınızı olasılıklı bir ölçekte kümeler halinde kümeler.

#Gözetimsiz (denetimsiz) öğrenme, modeli denetlemenize gerek olmayan bir makine öğrenme tekniğidir. Bunun yerine, modelin bilgileri keşfetmek için kendi başına çalışmasına izin vermeniz gerekir. Denetimsiz öğrenme algoritmaları, denetimli öğrenmeye kıyasla daha karmaşık işleme görevleri gerçekleştirmenizi sağlar.

#4.	Öğrenmesi istenen veri setinin bir ön işlemden geçiriilmesi doğru bir model oluşturulması için önemli olmaktadır. Bu anlamda öncelikle boş ve tekrarlı içerikler elimine edilir . Daha sonra ise ezberden kaçınılıp öğrenmesinin gerçekleştirilmesi için E-eğitim test ve validation oranının iyi belirlenmesi gerekmektedir. Bu nedenle eğitim + test veya eğitim + test + validation şeklinde ayrılmaktadır.  Genel olarak %70+%30 şeklinde ayrılsa da bu oran %(70±5)+%(30±5) şeklinde değişebilmektedir. Benzer şekilde diğer set eğitim(%60) + test (%20) + validation(%20) oranlarında kullanılmakta ve bu oran da kendi içinde değişiklik gösterebilmektedir. Ayrıca iyi bir eğitimin yapılarak veriş setindeki tüm elemanların homojen yararlanabilmesi için cross validation yöntemi 5 katsayısına uygun olarak yapılmaktadır. Bu yöntemde eğitim için ayrıcaln veri seti 5’e  bölünüp  bir tanesi validation için ayrılır ve her bir döngü (cycle) de eğitim ve validation yapılır ve validation yapılacak veri grubu 1 segment kaydırlır. Her bir segment validationa tabi tutulduktan sonra ortalama alınır ve  test yapılarak süreç tamamlanmış olur. Böylece testte elde edilen doğruluğu yüksek çıkması amaçlanır.

#5.	Öğrenmesi istenen veri setinin bir ön işlemden geçiriilmesi doğru bir model oluşturulması için önemli olmaktadır. Bu anlamda öncelikle boş ve tekrarlı içerikler elimine edilir . Uyumsuz, marjinal veya aykırı  verilen setten çıkarılabilir. Kaçırılan veya bozuk değerler mean veya median değerler ile doldurulabilir. 
#Bu bileşenler Python’ da sırasıyla null, dorpna ve IsolationForest araçlarıyla tespit edilip düzeltilebilirler.

#6.	Kısa süreli aralıklarla devamlılığı gerektiren veri setleri için continuous değişkenler olarak ele alınıp histogram eğrisi kullanılırken; sebze satışı gibi uzun aralıklar gerektiren veri setleri için discrete (ayrık) değişkenler olarak ele alınıp bu veri seti bar yapısı ile ifade edilebilir.

#7.	Şekilde gösterilen veri sürekli zamanlı veri olup; sıfr ve marjinal verilen içermektedir. Bu veriler elimine edildikten sonra makine öğrenmesi yapılması eğitilen verinin testi yaoıldığında daha yüksek doğrulukta bir başarım elde edilmesini sağlayacaktır.


