# python synthesize.py --restore_step 125000 \
#                      --dataset NgocHuyen \
#                      --mode single \
#                      --text "{k a6 k6 # b ie5 n5 - ts M5 N5 # s a1 # k uo3 # v ie1 m1 # v e1 # a1 # J M1 # b ie5 n5 - ts M5 N5 # d M92 N2 # t ie1 w1 - h wp5 a5 # sp J M1 # G 9X1 j1 # ie3 - ts aX3 j3 # sp}"


# python synthesize.py --restore_step 642000 \
#                      --dataset MultispeakerVbee \
#                      --mode batch \
#                      --speaker_id NgocHuyen \
#                      --source preprocessed_data/NgocHuyen/val.txt

# python synthesize.py --restore_step 324000 \
#                     --text "{t wp1 i1 # J ie1 n1 # sp 93 # N M92 j2 # dZ a2 # sp v ie8 k8 # ts u3 Nm3 # N M92 # k u5 m5 # k O5 - th e3 # i6 t6 # h ie7 w7 - k wp3 a3 # h 91 n1 # sp tS OX1 Nm1 # v ie8 k8 # N aX1 n1 - N M92 # b e7 J7 - t 9X8 t8 # sp J M1 N1 # l a2 m2 # dZ a3 m3 # m M6 k6 - d o7 # N ie1 m1 - tS OX7 Nm7 # k uo3 # b e7 J7 # sp t i3 - l e7 # m aX6 k6 # k a6 k6 # b ie5 n5 - ts M5 N5 # v a2 # t M3 # v OX1 Nm1 #}" \
#                      --dataset MultispeakerVbee_NgocHuyen \
#                      --mode single \
#                      --speaker_id NgocHuyen 

# """Multi-vocoder does not support batch inference yet"""
python synthesize.py --restore_step 642000 \
                    --text "{s OX1 Nm1 # k a2 N2 # l a3 N3 - ts EX5 J5 # sp d 91 n1 # th M1 # x ie5 w5 - n a7 j7 # J a2 - d 9X6 t6 # sp k a2 N2 # ts o2 Nm2 - ts 9X6 t6 #}" \
                     --dataset MultispeakerVbee \
                     --mode single \
                     --speaker_id NgocHuyen \
                     --multiple_vocoder