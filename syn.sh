# """Multi-vocoder does not support batch inference yet"""
# python synthesize.py --restore_step 792000 \
#                     --text "{th E1 w1 # J 9X7 n7 - d i7 J7 # k uo3 # k u8 kp8 # th u5 # i1 # sp N wp1 i1 - k 91 # z i8 k8 # k u5 m5 - dZ a1 - k 9X2 m2 # t ie6 p6 - t u8 kp8 # f a6 t6 - S i1 J1 # t a7 j7 # k a6 k6 # t i3 J3 # x a6 k6 # tS OX1 Nm1 # x u1 - v M8 k8 # n aX2 j2 # r 9X6 t6 # k a1 w1 # sp z O1 # k o1 Nm1 - t a6 k6 # k wp3 a3 n3 - l i5 # sp 9X6 p6 # n 93 # n uo1 j1 # m 95 j5 # th wp3 i3 - k 9X2 m2 # sp t a7 j7 # J ie2 w2 # d ie7 - f M91 N1 # k O2 n2 # l OX3 Nm3 - l E3 w3 # sp k o1 Nm1 - t a6 k6 # sp t ie1 m1 # f OX2 Nm2 # b o3 - S u1 Nm1 # ts O1 # d a2 n2 # v i8 t8 # th 92 j2 - v u7 # x o1 Nm1 # d M98 k8 # d a3 m3 - b a3 w3 #}" \
#                      --dataset MultispeakerVbee \
#                      --mode single \
#                      --speaker_id NgocHuyen \
#                      --multiple_vocoder

python synthesize.py --restore_step 792000 \
                    --text "{f a1 n1 - b e6 t6 # n aX2 j2 # d M98 k8 # l 9X8 p8 # sp k EX6 k6 # d 9X1 j1 # b a1 # n aX1 m1 # sp k O5 # h a1 j1 - ts aX1 m1 # n aX1 m1 - m M91 j1 # N i2 n2 # l a1 j1 # sp k u2 Nm2 # J ie2 w2 # d a1 - t a1 # x EX6 k6 - h a2 N2 #}" \
                     --dataset MultispeakerVbee \
                     --mode single \
                     --speaker_id NgocHuyen \
                     --multiple_vocoder \
                     --post_process